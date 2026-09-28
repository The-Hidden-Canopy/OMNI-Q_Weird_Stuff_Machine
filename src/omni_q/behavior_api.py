"""Narrow client seam between public OMNI-Q and private Hidden Canopy behavior logic.

The public repository sends only bounded, structured evidence.  The private Hub
service returns advisory dispositions such as REEXAMINE, SWITCH_PROVIDER, or
HOLD_GIVER.  It never grants execution authority: existing OMNI-Q mission,
skill, and supervisor gates remain authoritative.

No private behavior algorithm, thresholds, source code, or credentials belong in
this module.  Endpoint/key configuration is supplied at runtime.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping

BEHAVIOR_API_URL_ENV = "OMNIQ_BEHAVIOR_API_URL"
BEHAVIOR_API_KEY_ENV = "OMNIQ_BEHAVIOR_API_KEY"


@dataclass(frozen=True)
class EvidenceSource:
    source_id: str
    confidence: float
    freshness: float = 1.0
    continuity: float = 0.5
    provenance: float = 0.5
    independent: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "confidence": self.confidence,
            "freshness": self.freshness,
            "continuity": self.continuity,
            "provenance": self.provenance,
            "independent": self.independent,
        }


@dataclass(frozen=True)
class BehaviorEvidence:
    sources: tuple[EvidenceSource, ...] = ()
    contact_confidence: float | None = None
    target_stable: bool | None = None
    receiver_ready: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"sources": [source.as_dict() for source in self.sources]}
        if self.contact_confidence is not None:
            payload["contact_confidence"] = self.contact_confidence
        if self.target_stable is not None:
            payload["target_stable"] = self.target_stable
        if self.receiver_ready is not None:
            payload["receiver_ready"] = self.receiver_ready
        return payload


@dataclass(frozen=True)
class BehaviorAdvice:
    disposition: str
    reason: str = ""
    retry: bool = False
    reobserve: bool = False
    switch_provider: bool = False
    preserve_grip: bool = False
    confidence: Mapping[str, float] = field(default_factory=dict)
    request_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BehaviorAdvice":
        hints = value.get("hints") if isinstance(value.get("hints"), Mapping) else {}
        confidence = value.get("confidence") if isinstance(value.get("confidence"), Mapping) else {}
        return cls(
            disposition=str(value.get("disposition", "UNAVAILABLE")),
            reason=str(value.get("reason", "")),
            retry=bool(hints.get("retry", False)),
            reobserve=bool(hints.get("reobserve", False)),
            switch_provider=bool(hints.get("switch_provider", False)),
            preserve_grip=bool(hints.get("preserve_grip", False)),
            confidence={str(k): float(v) for k, v in confidence.items() if isinstance(v, (int, float))},
            request_id=str(value["request_id"]) if value.get("request_id") else None,
        )


class BehaviorAPIError(RuntimeError):
    pass


class BehaviorAPIClient:
    """Optional private behavior advisor.

    Failures are surfaced to the caller.  Callers should fail closed or fall
    back to existing public behavior; they must never treat an unavailable
    advisor as permission to broaden authority.
    """

    def __init__(self, url: str, *, key: str | None = None, timeout_s: float = 1.0) -> None:
        if not url.strip():
            raise ValueError("behavior API url must be non-empty")
        self.url = url.strip()
        self.key = key
        self.timeout_s = timeout_s

    @classmethod
    def from_env(cls) -> "BehaviorAPIClient | None":
        url = os.environ.get(BEHAVIOR_API_URL_ENV, "").strip()
        if not url:
            return None
        return cls(url, key=os.environ.get(BEHAVIOR_API_KEY_ENV))

    def advise(
        self,
        *,
        event: str,
        request_id: str,
        evidence: BehaviorEvidence,
        failure_kind: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> BehaviorAdvice:
        payload: dict[str, Any] = {
            "version": "omni-public-v1",
            "event": event,
            "request_id": request_id,
            "evidence": evidence.as_dict(),
            "context": dict(context or {}),
        }
        if failure_kind:
            payload["failure"] = {"kind": failure_kind}

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.key:
            headers["x-functions-key"] = self.key
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BehaviorAPIError(f"behavior advisor unavailable: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BehaviorAPIError("behavior advisor returned invalid JSON") from exc
        if not isinstance(data, Mapping):
            raise BehaviorAPIError("behavior advisor returned a non-object response")
        if data.get("advisory_only") is not True:
            raise BehaviorAPIError("behavior advisor response missing advisory_only=true")
        return BehaviorAdvice.from_mapping(data)
