"""Speechmatics realtime transport for the OMNI-Q voice boundary.

Vendor-specific code lives here, never in ``src/omni_q``.  ``src/omni_q/voice.py``
stays provider-neutral: it accepts plain payload mappings through
``SpeechmaticsRealtimeAdapter`` and imports no Speechmatics code at all.
"""
