"""Population trajectory sampling independent of dataset names or time units."""
import numpy as np


def stratified_candidates(candidates, spans, count, rng):
    if len(candidates) <= count:
        return list(candidates)
    order = np.argsort(np.asarray(spans, dtype=np.float64))
    strata = [chunk for chunk in np.array_split(order, 3) if len(chunk)]
    picked = []
    per_stratum = max(1, int(count) // max(1, len(strata)))
    for stratum in strata:
        take = min(per_stratum, len(stratum), count - len(picked))
        if take > 0:
            picked.extend(int(i) for i in rng.choice(stratum, size=take, replace=False))
    if len(picked) < count:
        remaining = np.asarray([i for i in range(len(candidates)) if i not in set(picked)], dtype=np.int64)
        if len(remaining):
            picked.extend(int(i) for i in rng.choice(remaining, size=min(count-len(picked), len(remaining)), replace=False))
    return [candidates[i] for i in picked[:count]]


def training_pairs(times, sources_per_target, rng):
    times = np.asarray(times, dtype=np.float64)
    pairs = []
    for target in range(len(times)):
        earlier = [source for source in range(len(times)) if times[source] < times[target]]
        later = [source for source in range(len(times)) if times[source] > times[target]]
        count = min(sources_per_target, len(earlier) + len(later))
        n_earlier = min(len(earlier), count // 2)
        n_later = min(len(later), count - n_earlier)
        remaining = count - n_earlier - n_later
        extra = min(remaining, len(earlier) - n_earlier)
        n_earlier += extra
        n_later += min(remaining-extra, len(later)-n_later)
        selected = stratified_candidates(earlier, [abs(times[s]-times[target]) for s in earlier], n_earlier, rng)
        selected += stratified_candidates(later, [abs(times[s]-times[target]) for s in later], n_later, rng)
        pairs.extend((int(source), target) for source in selected)
    return pairs


def endpoint_triples(times, count, rng):
    times = np.asarray(times, dtype=np.float64)
    candidates = [(a, b, target) for target in range(len(times))
                  for a in range(len(times)) if a != target and times[a] != times[target]
                  for b in range(a+1, len(times)) if b != target and times[b] != times[target]]
    spans = [max(abs(times[target]-times[a]), abs(times[target]-times[b])) for a, b, target in candidates]
    return stratified_candidates(candidates, spans, count, rng)
