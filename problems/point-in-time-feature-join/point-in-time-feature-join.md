## Why training data needs a clock

A model learns from rows that pair a situation with an outcome. If the situation is described with numbers that were only computed after the outcome happened, the model sees the answer hidden inside the question. It will look excellent during offline evaluation and then disappoint in production, where the future is not available. The leakage topic of the course calls this time leakage, and the most common way it enters a training set is through features that were refreshed after the event they describe.

A point-in-time join prevents this. For every event, it looks back in time and picks the feature value that a serving system would have had in its hands at that moment, and nothing newer is allowed in.

## The two timestamps

Every feature row carries the time at which its value became available. Every event carries the time at which the prediction would have been made. The join compares the two. A feature row qualifies for an event when its timestamp is at or before the event timestamp. Among the qualifying rows for the same entity, the newest one is the value that serving would have returned, so that is the value the training row receives.

Think of a bank card. Your balance changes several times a day. If a fraud model scored a purchase at 14:05, the balance it could have used was the one after the last update before 14:05. By the time someone builds the training set the balance has moved several times since, and none of those later values belong in the row.

## The time to live window

Feature stores also attach a maximum age to each feature, often called a time to live or TTL. A seven day spending average that was last refreshed two months ago is not a useful description of the user today, and a serving system would treat it as missing. The join mirrors that rule. A row that is older than the window, measured from the event, is dropped even though it is in the past. When nothing within the window qualifies, the feature value is missing, and the training pipeline fills it the same way serving would, for example with a default.

The window is inclusive at both ends in this problem. A row at exactly the event time counts, and a row exactly ttl units old counts.

## Why the window matters for consistency

The feature consistency topic explains that training and serving must compute the same thing. If serving refuses to use a feature older than a day, but the training join happily uses a value from last month, the model trains on a feature that behaves differently from the one it will see live. Giving the join the same window as serving keeps the two in step.

## Worked numbers

Take ttl equal to 10 and an entity with feature rows at times 3, 8 and 20. An event at time 12 can see the rows at 3 and 8. Both are in the past and both are within 10 units, so the newer one, at time 8, wins.

An event at time 18 sees only the row at 8, because 18 minus 3 is 15, which is outside the window, while 18 minus 8 is exactly 10, which is still inside it. An event at time 19 sees nothing, because the row at 8 is now 11 units old and the row at 20 is still in the future. Moving the event by a single unit changed the training row from a value to a missing value, which is why careful engineers test the boundary of the window rather than trusting it.

## How it comes up in interviews

A typical question is how you would build the training set for a fraud model when the features live in a feature store. An answer a newcomer can give aloud runs like this. I would join each labelled transaction to the feature rows for the same card using the feature timestamps. For each transaction I would take the newest row written before it and no older than the time to live that serving uses, which for a one day spending total I would set to one day. That gives the training set the same view of each feature that serving had, at the cost of some rows receiving a missing value where serving would also have seen one.

A follow up often asks how you would notice that a training set was built without this join, and the honest answer is that the offline metrics look better than the live ones, because the model trained with hindsight.