## Why teams throw away negatives

In click prediction, fraud detection and many other problems, the interesting class is rare. A training set with a billion rows where only a million are positive spends almost all of its storage and compute on examples that teach the model very little new. A common remedy keeps every positive and only a sampled fraction of the negatives, so a billion rows becomes a hundred million and the training job finishes in an hour instead of a day.

The sampling topic of the course accepts this trade, and then adds a warning that this problem is about. The model trained on the smaller set has learned a different world, one where positives are far more common than they are in production.

## What the model learned instead

Suppose one in a thousand impressions is clicked, and the pipeline keeps one negative in ten. In the training set there is now one click for every hundred non-clicks, so the base rate the model sees is about one percent instead of one tenth of a percent. A well fitted model matches the base rate it was shown, so its average score is around 0.01 when the population rate is around 0.001. Every score is inflated, and the inflation is not a constant shift. Scores near zero and one move the least and scores in the middle move the most.

Think of a survey that interviewed ten times as many city residents as the country has. Any percentage read straight off the survey is wrong for the country, but the raw interview answers are still perfectly good once each group is given its proper weight.

## The correction

The fix multiplies the odds back by the keep rate. In a probability of p, the odds of a positive are p over one minus p. The sampling increased those odds by a factor of one over w, where w is the keep rate, so the population odds are w times the sampled odds. Turning odds back into a probability gives

$$
q = \frac{p}{p + \frac{1 - p}{w}}
$$

which is the formula in this problem. It comes from the practical lessons paper on predicting ad clicks and has been standard in advertising systems since. With w equal to 1 the formula returns p unchanged, and with p equal to 0 or 1 the formula returns 0 or 1, because there is nothing to reweight at the extremes.

## What the correction does not fix

The formula undoes exactly one thing, the change in the ratio of positives to negatives that the sampling introduced. It does not know whether the model was poorly calibrated for other reasons, such as too little training or a shift in the data since training. Those problems need a calibration step, such as isotonic regression, which the catalog covers separately. A useful habit is to apply the sampling correction first and then measure calibration error on held out data that was not downsampled. If the error is still large, something other than the sampling is wrong.

## Why ranking does not care but bidding does

The correction is a monotone transformation, which means a higher raw score always becomes a higher corrected score. A system that only sorts candidates by score, such as a ranker choosing which ten items to show, produces the same order either way. A system that compares the score against a fixed threshold, or multiplies it by a value to compute a bid, reads the score as a probability and is misled by the inflation. For an advertiser paying per click, a tenfold overestimate of the click probability is a tenfold overestimate of what the impression is worth.

## How it comes up in interviews

When a candidate proposes keeping one negative in ten to shrink a click data set, the interviewer usually asks what that does to the predicted probabilities. A newcomer's answer runs like this. The model will see clicks as ten times more common than they are, so its scores are inflated. I would correct each score by dividing p by p plus one minus p over the keep rate before any bidding logic reads it. Ranking does not need the correction because the order of the scores does not change, but the bid does, since a tenfold overestimate of the click probability is a tenfold overbid.

A follow up sometimes asks how to choose the keep rate. A lower rate saves compute while making the model noisier, so I would tune it against log loss on a validation set that was not downsampled.