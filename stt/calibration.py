"""Microphone calibration analysis: samples -> suggested settings.

Pure statistical reduction of collected calibration samples (noise/speech
energies, silence durations, VAD probabilities) into suggested energy
threshold, phrase timeout, and VAD settings. Extracted from speech_to_text.py
so it is importable and unit-testable. Stdlib-only: the one numpy percentile
call is replaced by `_percentile` below, which reproduces numpy's default
linear-interpolation method exactly, so results are unchanged.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List


def _percentile(values: List[float], pct: float) -> float:
    """The ``pct``-th percentile of ``values`` using linear interpolation.

    Matches numpy.percentile's default ('linear') method so the extraction is
    behavior-preserving. ``values`` need not be sorted; must be non-empty.
    """
    xs = sorted(values)
    if not xs:
        raise ValueError("percentile of empty sequence")
    k = (len(xs) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(xs[int(k)])
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def analyze_calibration_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze calibration data and suggest optimal settings."""
    results: Dict[str, Any] = {
        "suggestions": {},
        "analysis": {},
        "confidence": "medium"
    }

    # 1. ANALYZE ENERGY THRESHOLD
    # With two-step calibration, we ALWAYS have noise data from step 1
    if data["noise_samples"]:
        noise_energies = [s["energy"] for s in data["noise_samples"]]
        speech_energies = [s["energy"] for s in data["speech_samples"]] if data["speech_samples"] else []

        avg_noise = statistics.mean(noise_energies)
        max_noise = max(noise_energies)
        # Use 75th percentile to ignore outlier noise spikes
        noise_75th = _percentile(noise_energies, 75) if len(noise_energies) > 10 else max_noise

        if speech_energies:
            # Have both noise and speech - optimal case (two-step calibration success)
            min_speech = min(speech_energies)
            avg_speech = statistics.mean(speech_energies)

            # Set threshold between noise ceiling and speech floor
            # Use 75th percentile of noise to ignore spikes
            suggested_threshold = int((noise_75th + min_speech) / 2)

            # Ensure it's above noise but below speech
            suggested_threshold = max(int(noise_75th * 1.2), suggested_threshold)
            suggested_threshold = min(int(min_speech * 0.9), suggested_threshold)

            results["analysis"]["speech_level"] = {
                "minimum": round(min_speech, 1),
                "average": round(avg_speech, 1),
                "samples": len(speech_energies)
            }
            results["analysis"]["threshold_confidence"] = "high"
        else:
            # Only have noise from step 1 (user didn't speak in step 2)
            # Use conservative threshold based on average noise
            suggested_threshold = int(avg_noise * 2.0)
            suggested_threshold = max(300, suggested_threshold)

            # Add warning about missing speech samples
            if "warnings" not in results:
                results["warnings"] = []
            results["warnings"].append(
                "No speech detected in Step 2. "
                "Using conservative threshold based on noise floor only."
            )

            # Mark threshold confidence as low
            results["analysis"]["threshold_confidence"] = "low"

        # Clamp to reasonable range
        suggested_threshold = max(100, min(10000, suggested_threshold))

        results["suggestions"]["energy_threshold"] = suggested_threshold
        results["analysis"]["noise_level"] = {
            "average": round(avg_noise, 1),
            "maximum": round(max_noise, 1),
            "percentile_75": round(noise_75th, 1),
            "environment": "quiet" if avg_noise < 500 else "normal" if avg_noise < 2000 else "noisy"
        }

    # 2. ANALYZE PHRASE TIMEOUT
    if data["silence_durations"]:
        silence_durations = data["silence_durations"]

        # DEBUG: Log raw silence durations data
        print(f"[CALIBRATION-ANALYSIS] Analyzing {len(silence_durations)} silence durations: {silence_durations[:20] if len(silence_durations) > 20 else silence_durations}", flush=True)

        # Find typical pause length (median of shorter pauses)
        short_pauses = [d for d in silence_durations if d < 5.0]  # Ignore very long pauses

        print(f"[CALIBRATION-ANALYSIS] After filtering (< 5.0s): {len(short_pauses)} short pauses", flush=True)

        if short_pauses:
            median_pause = statistics.median(short_pauses)

            print(f"[CALIBRATION-ANALYSIS] Median pause: {median_pause:.2f}s", flush=True)

            # Suggest phrase_timeout slightly above median pause
            # This splits on longer pauses but not normal speech pauses
            suggested_timeout = round(median_pause * 1.3, 1)
            suggested_timeout = max(1.0, min(5.0, suggested_timeout))  # Clamp to 1-5 seconds

            print(f"[CALIBRATION-ANALYSIS] Suggested phrase_timeout: {suggested_timeout}s", flush=True)

            results["suggestions"]["phrase_timeout"] = suggested_timeout
            results["analysis"]["pause_pattern"] = {
                "median_pause": round(median_pause, 2),
                "min_pause": round(min(short_pauses), 2),
                "max_pause": round(max(short_pauses), 2),
                "total_pauses": len(silence_durations)
            }

    # 3. ANALYZE VAD SETTINGS
    if data["vad_probabilities"]:
        vad_probs = data["vad_probabilities"]
        avg_vad = statistics.mean(vad_probs)
        min_vad = min(vad_probs)

        # If average VAD confidence is high, can use stricter threshold
        if avg_vad > 0.8:
            suggested_vad_threshold = 0.6  # Stricter
        elif avg_vad > 0.6:
            suggested_vad_threshold = 0.5  # Normal
        else:
            suggested_vad_threshold = 0.4  # More lenient

        results["suggestions"]["vad_threshold"] = suggested_vad_threshold
        results["suggestions"]["vad_enabled"] = True
        results["analysis"]["vad_performance"] = {
            "average_confidence": round(avg_vad, 2),
            "minimum_confidence": round(min_vad, 2),
            "recommendation": "strict" if avg_vad > 0.8 else "normal" if avg_vad > 0.6 else "lenient"
        }

    # 4. DETERMINE CONFIDENCE LEVEL
    speech_samples = len(data.get("speech_samples", []))
    noise_samples = len(data.get("noise_samples", []))

    if speech_samples >= 10 and noise_samples >= 10:
        results["confidence"] = "high"
    elif speech_samples >= 5 or noise_samples >= 5:
        results["confidence"] = "medium"
    else:
        results["confidence"] = "low"

    return results
