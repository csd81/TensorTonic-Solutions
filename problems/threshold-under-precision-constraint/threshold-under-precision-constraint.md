## A score is not a decision

A classifier produces a number, and someone turns that number into an action by comparing it to a threshold. Every threshold trades two mistakes against each other. Raise it and fewer innocent transactions are blocked, but more fraud slips through. Lower it and more fraud is caught, at the price of annoying honest customers. The classification metrics topic of the course describes precision and recall as the two numbers that track this trade, and it says that a model should be compared at a meaningful threshold rather than at whatever default the library used.

## Precision and recall at one threshold

Fix a threshold and call every row above it positive. Precision asks what share of the rows we flagged were positives. Recall asks what share of the positives we flagged. The two move in opposite directions as the threshold slides, because flagging more rows can only add true positives or false positives, never remove either.

An everyday picture is a smoke detector. A very sensitive one catches every fire and also every piece of burnt toast. A very insensitive one never bothers you about toast and sometimes misses a fire. Which one you want depends on whether you live in a wooden house or a kitchen.

## Stating the rule as a constraint

Businesses rarely say they want to maximise a blend of precision and recall. They say things like the fraud team can review at most two hundred cases a day, or no more than one in ten blocked payments may be legitimate. The second kind of rule is a floor on precision. Once the floor is fixed, the remaining goal is simple, catch as much as possible, which means pick the threshold with the highest recall among those that respect the floor.

The reverse constraint also appears. A medical screening test may need recall of at least 99 percent, with precision as high as that allows. The same search works with the roles swapped.

## Why only the observed scores need checking

Moving the threshold between two consecutive observed scores changes nothing, because no row crosses from one side to the other. So the full set of distinct behaviours is reached by trying each distinct score as the threshold, which turns a search over a continuous line into a loop over at most as many candidates as there are rows. Tied scores are handled naturally, since all rows with the same score move together.

## Ties and the shape of the search

As the threshold falls, recall can only rise or stay the same. Two thresholds with equal recall differ only in how many negatives they also admit, so the higher one has better precision. When recall ties, preferring the higher precision therefore means preferring the stricter threshold. This matters because the constraint is a floor, and a threshold sitting exactly on the floor leaves no room for the small shifts in score distribution that happen in production. The stricter choice gives a little margin for free.

## What can go wrong

If the highest scoring row is a negative, the very first candidate already has a precision of zero, and if the floor is high enough nothing will ever meet it. Returning None in that case is the honest answer, and a serving system should treat it as a signal that the model cannot satisfy the business rule and that something, the model or the rule, has to change. Returning the best failing threshold instead would ship a system that violates its own promise.

## How it comes up in interviews

After a candidate names precision and recall as the metrics for a fraud system, the interviewer almost always asks which threshold they would use and how they would choose it. A newcomer's answer runs like this. I would ask the fraud team what precision they can live with, say 90 percent so that at most one blocked payment in ten is legitimate. Then I would evaluate every observed score as a threshold on held out data and pick the one with the highest recall among those that meet the floor. When two thresholds tie on recall I would take the stricter one because it keeps a little margin above the floor.

A follow up often asks what happens after retraining, and the answer is that the new model's scores rarely line up with the old threshold, so the search has to be repeated for each new model.