# Editorial Quality Evaluation

Milestone 7 measures Gold Miner against moments independently selected by human editors. It does not call transcription, model, embedding, or other paid providers.

## Dataset format

Create a versioned JSON file with a dataset name and one or more episodes. Each episode contains an ID, a ranking path, and human-selected gold moments. Every gold moment records an ID, start and end milliseconds, relevance from 1 to 3, and an idea group. Give alternate phrasings of the same underlying idea the same idea group. Ranking paths are resolved relative to the dataset file.

See tests/fixtures/evaluation_dataset.json for a complete example.

## Matching rule

A prediction matches a human moment when their timestamp intersection covers at least 50 percent of the shorter span. This tolerates reasonable boundary differences while preventing a broad episode section from matching unrelated moments.

## Metrics

- Recall at 10 measures how many human gold moments appear in the first ten predictions.
- Precision at 5 measures how many of the first five predictions match a human gold moment.
- NDCG at 10 rewards placing higher-relevance moments earlier.
- Boundary error is the mean absolute start and end error for matched predictions, in seconds.
- Duplicate rate is the share of the first ten predictions that redundantly match an already represented idea group.
- Run reliability is the share of dataset episodes whose ranking file loads and evaluates successfully.

## Run and record a baseline

Run:

    goldminer-evaluate path/to/editorial_dataset.json --output evaluations/baseline-v1.json

After an algorithm change:

    goldminer-evaluate path/to/editorial_dataset.json --baseline evaluations/baseline-v1.json --output evaluations/proposed.json --fail-on-regression

The command exits with status 1 when a metric regresses. Use the max regression option only when the team has explicitly agreed to the tradeoff.

## Human review process

For each episode, a senior editor should watch the source independently before seeing Gold Miner results, record every publishable moment with exact boundaries, assign relevance, and group duplicate ideas. Keep the dataset fixed while comparing algorithm versions. Add new episodes as a new dataset version rather than silently changing an established benchmark.

The synthetic dataset and baseline under tests/fixtures prove metric and regression behavior only. They are not evidence of real editorial quality. A production baseline requires approximately 20 to 50 independently reviewed episodes, as specified in the product document.
