"""サーバー内で観測済みのユーザー名キャッシュと、公開回答の名前解決。"""

import heapq
import json
import re
import unicodedata
from itertools import zip_longest

from sqlmodel import Session, select

from .db import DiscordUserRecord, get_engine

_ID = re.compile(r"[0-9]{5,25}")
_MARKER = re.compile(r"member-([0-9]{5,25})|<@!?([0-9]{5,25})>")
_USER_FIELD = re.compile(r'"(?:author_id|user_id|owner_id|member_id)"\s*:\s*"([0-9]{5,25})"')


def sdk_user(user) -> dict:
    data = {"id": str(user.id)}
    if hasattr(user, "name"):
        data["username"] = user.name
    if hasattr(user, "global_name"):
        data["global_name"] = user.global_name
    if hasattr(user, "nick"):
        data["nick"] = user.nick
    return data


def message_users(messages) -> list[dict]:
    result = []
    for message in messages:
        author = dict(message.get("author") or {})
        member = message.get("member")
        if isinstance(member, dict) and "nick" in member:
            author["nick"] = member["nick"]
        result.append(author)
    return result


def remember_users(guild_id, users):
    pending = {}
    for user in users:
        user_id = str(user.get("id", ""))
        if _ID.fullmatch(user_id):
            pending.setdefault(user_id, {}).update(user)
    if not pending:
        return
    guild = str(guild_id)
    ids = list(pending)
    with Session(get_engine()) as session:
        existing = {}
        for offset in range(0, len(ids), 256):
            keys = [f"{guild}:{user_id}" for user_id in ids[offset : offset + 256]]
            existing.update(
                {
                    row.user_id: row
                    for row in session.exec(select(DiscordUserRecord).where(DiscordUserRecord.user_key.in_(keys)))
                }
            )
        for user_id, user in pending.items():
            row = existing.get(user_id) or DiscordUserRecord(
                user_key=f"{guild}:{user_id}", guild_id=guild, user_id=user_id
            )
            for source, field in (("username", "username"), ("global_name", "global_name"), ("nick", "nickname")):
                if source in user:
                    name = " ".join(str(user[source] or "").split())[:100]
                    setattr(row, field, name or (None if source == "nick" else ""))
            session.add(row)
        session.commit()


def cached_users(guild_id, user_ids=None) -> list[dict]:
    query = select(DiscordUserRecord).where(DiscordUserRecord.guild_id == str(guild_id))
    if user_ids is not None:
        if not user_ids:
            return []
        query = query.where(DiscordUserRecord.user_key.in_([f"{guild_id}:{user_id}" for user_id in user_ids]))
    with Session(get_engine()) as session:
        return [
            {
                "user_id": row.user_id,
                "display_name": row.nickname or row.global_name or row.username or None,
                "aliases": list(dict.fromkeys(name for name in (row.nickname, row.global_name, row.username) if name)),
            }
            for row in session.exec(query)
        ]


def resolve_users(guild_id, user_ids) -> dict:
    wanted = set(user_ids)
    found = {user["user_id"]: user for user in cached_users(guild_id, wanted)}
    return {
        "source": "cache",
        "users": [
            {
                "user_id": user_id,
                "display_name": found.get(user_id, {}).get("display_name"),
                "resolved": bool(found.get(user_id, {}).get("display_name")),
            }
            for user_id in dict.fromkeys(user_ids)
        ],
    }


def normalized(name: str) -> str:
    return unicodedata.normalize("NFKC", name).casefold().strip()


def search_users(guild_id, query: str, top_k: int) -> dict:
    needle = normalized(query)
    candidates = []
    for user in cached_users(guild_id):
        if not user["aliases"]:
            continue
        distance = min(sum(a != b for a, b in zip_longest(needle, normalized(alias))) for alias in user["aliases"])
        candidates.append({**user, "distance": distance, "exact": distance == 0})
    selected = heapq.nsmallest(
        top_k, candidates, key=lambda row: (row["distance"], normalized(row["display_name"]), int(row["user_id"]))
    )
    return {
        "source": "cache",
        "metric": "hamming_with_length_penalty",
        "candidates": selected,
        "ambiguous": bool(selected) and sum(row["distance"] == selected[0]["distance"] for row in candidates) > 1,
    }


def user_ids_in(value) -> set[str]:
    result = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"author_id", "user_id", "owner_id", "member_id"} and _ID.fullmatch(str(item)):
                result.add(str(item))
            elif key == "author" and isinstance(item, dict) and _ID.fullmatch(str(item.get("id", ""))):
                result.add(str(item["id"]))
            result.update(user_ids_in(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            result.update(user_ids_in(item))
    elif isinstance(value, str):
        if value.startswith("data:image/"):
            return result
        result.update(match.group(1) or match.group(2) for match in _MARKER.finditer(value))
        result.update(_USER_FIELD.findall(value))
        if value.startswith(("{", "[")):
            try:
                result.update(user_ids_in(json.loads(value)))
            except (ValueError, RecursionError):
                pass
    return result


def public_user_names(text: str, guild_id, *, sources=()) -> str:
    known = user_ids_in(sources) | user_ids_in(text)
    candidates = set(re.findall(r"(?<![0-9])[0-9]{5,25}(?![0-9])", text)) | known
    names = {user["user_id"]: user["display_name"] for user in cached_users(guild_id, candidates)}
    known.update(names)

    def name(user_id):
        value = names.get(user_id) or "あるメンバー"
        value = _MARKER.sub("あるメンバー", value)
        return re.sub(r"(?<![0-9])[0-9]{5,25}(?![0-9])", lambda m: "あるメンバー" if m[0] in known else m[0], value)

    text = _MARKER.sub(lambda match: name(match.group(1) or match.group(2)), text)
    return re.sub(
        r"(?<![0-9])[0-9]{5,25}(?![0-9])", lambda match: name(match[0]) if match[0] in known else match[0], text
    )
