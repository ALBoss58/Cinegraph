# Contributing

1. `pip install -e .[ai,full,dev]` then `python scripts/fetch_test_assets.py && python examples/make_test_media.py`.
2. `python -m pytest -q tests` must stay green.
3. New vision/audio features need a ground-truth test (synthetic clip with known answer + a measured metric),
   not just "it runs". See tests/test_v02.py, tests/test_ai.py.
4. Images inside the engine are premultiplied float32 RGBA 0..1; never cache a property's default; keep randomness
   seeded per layer+frame so parallel segments match.
