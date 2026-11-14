## Why a random split lies about the future

A validation set exists to estimate how a model will do on data it has not seen. When rows are ordered in time, the data the model will not have seen is the data that comes later.

A random split ignores this. It trains on rows from Tuesday and Thursday and validates on Wednesday, so the model is asked to interpolate between days it already knows. Production asks it to extrapolate into next week. The score from the random split is therefore an optimistic estimate, and sometimes a wildly optimistic one.

The leakage topic of the course describes this as making the split match the future use. If the model will be trained on the past and applied to the future, the validation set must sit entirely after the training set.

## The cutoff

A time-based split chooses a cutoff. Everything before it is training material and everything after it, up to some end point, is validation material. The end point matters too, because rows far in the future may belong to a later test set or may be incomplete. Keeping the validation window bounded keeps the comparison honest.

Here is an everyday picture. A weather forecaster who wants to know how good her model is at predicting Friday from Monday to Thursday should check it on past weeks in exactly that order. Hiding a random Wednesday and guessing it from its neighbours tests a different skill.

## Why the gap exists

Many features and labels are computed over windows of time. A seven day rolling average that ends on the cutoff day contains only information from before the cutoff, which is fine. A label that says whether a user churned in the next thirty days reaches thirty days past the row's own timestamp. If a training row is dated one day before the cutoff, its label was decided by events that happen inside the validation window. The model sees those events through the label, and the validation score is contaminated.

A gap, sometimes called an embargo or a purge window, removes the rows closest to the cutoff on the training side so that no training label can reach into the validation period. Its length should match the longest window used in any feature or label. With thirty day churn labels, the gap is thirty days. With a label that is known immediately, the gap can be zero.

## Half open intervals

This problem uses intervals that include their start and exclude their end. Training covers timestamps up to but not including train_end. The gap covers train_end up to but not including train_end plus gap. Validation covers train_end plus gap up to but not including val_end. Half open intervals fit together without overlap and without missing a single integer, which is why time libraries and database queries use them so often.

The example with gap equal to zero shows the consequence. The row at exactly train_end is not in training, and since the gap has no width, it is the first validation row.

## What this costs

Rows inside the gap are thrown away, and so are rows after val_end. On a small data set that loss hurts. Teams accept it because the alternative is a validation score they cannot trust, and a trustworthy but slightly noisier score is worth more to them than one that is precise and wrong. When the data set is large enough, the same idea is repeated several times with the cutoff moving forward, which the course mentions as rolling or expanding window validation.

## How it comes up in interviews

A common interview question describes a churn model whose validation accuracy was 92 percent and whose live accuracy is 70 percent, and asks what went wrong. A newcomer's answer runs like this. I would first check how the validation set was built, because a random split lets the model validate on days it already trained on. I would cut the data at a date instead and train on everything before it. Because the churn label looks thirty days ahead, I would leave a thirty day gap after the cutoff before the validation window starts, accepting that the rows in those thirty days are lost.

A follow up usually asks how long the gap should be, and the answer is the longest horizon that any feature or label looks across.