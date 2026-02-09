# pyribs 0.9+ API Migration Notes

This document captures breaking API changes discovered when migrating from older pyribs versions to 0.9+.

## CVTArchive Initialization

### `cells` parameter deprecated
```python
# OLD (pre-0.9)
archive = CVTArchive(
    solution_dim=1,
    cells=1000,
    ranges=ranges,
    seed=42,
)

# NEW (0.9+) - must pre-compute centroids
from scipy.cluster.vq import kmeans

rng = np.random.default_rng(seed)
num_samples = num_cells * 100
samples = rng.uniform(
    low=[r[0] for r in ranges],
    high=[r[1] for r in ranges],
    size=(num_samples, len(ranges)),
)
centroids, _ = kmeans(samples, num_cells)

archive = CVTArchive(
    solution_dim=1,
    ranges=ranges,
    centroids=centroids,  # Required now
)
```

## Archive.add() Return Value

### Returns dict instead of array
```python
# OLD (pre-0.9)
status_batch = archive.add(solution=..., objective=..., measures=...)
# status_batch was array of AddStatus

# NEW (0.9+)
result = archive.add(solution=..., objective=..., measures=...)
# result is dict: {'status': array([...]), 'value': array([...])}
status_array = result["status"]
```

## AddStatus Enum

### `AddStatus.IMPROVE` removed
```python
# OLD (pre-0.9)
from ribs.archives import AddStatus
if status == AddStatus.NEW:
    ...
if status == AddStatus.IMPROVE:
    ...

# NEW (0.9+) - use integer comparison
# NOT_ADDED = 0, NEW = 1, IMPROVE = 2
if int(status) == 1:  # NEW
    ...
if int(status) == 2:  # IMPROVE
    ...
if int(status) > 0:   # Was added (NEW or IMPROVE)
    ...
```

## Archive Attributes Removed

### `archive.occupied` removed
```python
# OLD (pre-0.9)
occupied_indices = np.where(archive.occupied)[0]
if archive.occupied[index]:
    ...

# NEW (0.9+) - iterate directly over archive
for elite in archive:
    # elite is a dict
    ...
```

### `archive.index` removed
```python
# OLD (pre-0.9)
occupied_indices = np.array(list(archive.index))
if index in archive.index:
    ...

# NEW (0.9+) - iterate directly
for elite in archive:
    idx = elite["index"]
    ...
```

### `archive.ranges` removed
```python
# OLD (pre-0.9)
measure_dim = len(archive.ranges)

# NEW (0.9+)
measure_dim = archive.measure_dim
```

### `archive.data("field")` changed
```python
# OLD (pre-0.9)
objectives = archive.data("objective")[indices]
solutions = archive.data("solution")[indices]

# NEW (0.9+) - iterate over archive instead
for elite in archive:
    obj = elite["objective"]
    sol = elite["solution"]
    measures = elite["measures"]
    idx = elite["index"]
```

## Iterating Over Archive

### Elites are dicts, not named tuples
```python
# OLD (pre-0.9) - might have been named tuples
for elite in archive:
    obj = elite.objective
    sol = elite.solution

# NEW (0.9+) - elites are dicts
for elite in archive:
    obj = elite["objective"]
    sol = elite["solution"]
    measures = elite["measures"]
    idx = elite["index"]
    threshold = elite["threshold"]
```

## EvolutionStrategyEmitter.tell()

### Requires `solution` and `add_info` arguments
```python
# OLD (pre-0.9)
emitter.tell(
    objective=objectives,
    measures=measures,
)

# NEW (0.9+)
add_info = archive.add(
    solution=solutions,
    objective=objectives,
    measures=measures,
)
emitter.tell(
    solution=solutions,
    objective=objectives,
    measures=measures,
    add_info=add_info,
)
```

## Visualization

### `cvt_archive_heatmap` only works for 1D/2D
```python
# Check dimensionality before using
if archive.measure_dim <= 2:
    cvt_archive_heatmap(archive, ax=ax)
else:
    # Use custom 3D visualization instead
    pass
```

## Summary of Key Changes

| Old API | New API (0.9+) |
|---------|----------------|
| `CVTArchive(cells=N)` | `CVTArchive(centroids=...)` |
| `archive.add()` returns array | Returns dict with `"status"` key |
| `AddStatus.IMPROVE` | Use `int(status) == 2` |
| `archive.occupied` | Iterate with `for elite in archive` |
| `archive.index` | Iterate with `for elite in archive` |
| `archive.ranges` | `archive.measure_dim` |
| `archive.data("field")` | Iterate and access `elite["field"]` |
| `elite.solution` | `elite["solution"]` (dict access) |
| `emitter.tell(obj, meas)` | `emitter.tell(sol, obj, meas, add_info)` |
