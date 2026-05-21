from __future__ import annotations

"""Coverage provider interfaces for Stage 4.

Native coverage needs source builds and toolchain-specific artifacts, so the
core runner treats it as a provider capability instead of as library logic.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Protocol, Sequence


class CoverageProvider(Protocol):
    name: str

    def available(self) -> bool:
        ...

    def limitations(self) -> Sequence[str]:
        ...

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def export(self, output_dir: str) -> Dict[str, Any]:
        ...


@dataclass
class NullCoverageProvider:
    name: str = "none"
    reason: str = "coverage provider unavailable"

    def available(self) -> bool:
        return False

    def limitations(self) -> Sequence[str]:
        return [self.reason]

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def export(self, output_dir: str) -> Dict[str, Any]:
        return {"available": False, "provider": self.name, "reason": self.reason}


@dataclass
class PythonCoverageProvider:
    source: Sequence[str] = field(default_factory=list)
    omit: Sequence[str] = field(default_factory=list)
    name: str = "python_coverage"

    def available(self) -> bool:
        try:
            import coverage  # noqa: F401
        except Exception:
            return False
        return True

    def limitations(self) -> Sequence[str]:
        return ["Python coverage measures Python package/wrapper lines, not native kernels."]

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def export(self, output_dir: str) -> Dict[str, Any]:
        return {
            "available": self.available(),
            "provider": self.name,
            "output_dir": output_dir,
            "source": list(self.source),
            "omit": list(self.omit),
            "limitations": list(self.limitations()),
        }


@dataclass
class NativeCoverageProvider:
    source_root: str = ""
    build_dir: str = ""
    name: str = "native"

    def available(self) -> bool:
        return bool(self.source_root and self.build_dir and Path(self.source_root).exists() and Path(self.build_dir).exists())

    def limitations(self) -> Sequence[str]:
        if self.available():
            return []
        return ["Native coverage requires an instrumented source build and coverage artifacts."]

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def export(self, output_dir: str) -> Dict[str, Any]:
        return {
            "available": self.available(),
            "provider": self.name,
            "source_root": self.source_root,
            "build_dir": self.build_dir,
            "limitations": list(self.limitations()),
        }


@dataclass
class GcovLcovNativeCoverageProvider(NativeCoverageProvider):
    gcovr_executable: str = "gcovr"
    name: str = "native_gcov_lcov"

    def available(self) -> bool:
        return super().available() and any(Path(self.build_dir).rglob("*.gcno"))

    def limitations(self) -> Sequence[str]:
        if self.available():
            return []
        return ["GCOV/LCOV native coverage requires gcovr plus .gcno/.gcda files from an instrumented build."]

