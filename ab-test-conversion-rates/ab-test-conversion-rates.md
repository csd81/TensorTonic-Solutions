## Two numbers are never exactly equal

Run the same checkout page for two groups of ten thousand people and the two conversion rates will differ even though the pages are identical. Random chance alone moves a rate up or down by a fraction of a percent. An experiment is therefore a question about whether the gap between two groups is bigger than chance would produce on its own, and a statistical test is the tool that answers it.

The experiments topic of the course asks you to report two things together, how large the effect is and how uncertain you are about it. The test statistic and p-value measure the uncertainty, and the confidence interval shows both at once.

## The lift

The simplest summary is the lift, the treatment rate minus the control rate. If control converts at twelve percent and treatment at thirteen, the lift is one percentage point. This absolute difference is the number a product team can turn into revenue, and it is the center of the confidence interval.

## The null hypothesis and the pooled rate

The test imagines a world where the new page changes nothing, so both groups share a single true conversion rate. Our best guess for that shared rate is the pooled rate, which is all conversions divided by all users. Under that assumption the lift we observed should hover around zero with a spread given by the pooled standard error,

$$
\mathrm{SE}_{pooled} = \sqrt{\bar p (1 - \bar p) \left( \frac{1}{n_c} + \frac{1}{n_t} \right)}
$$

where the bar denotes the pooled rate and the two n values are the group sizes. Dividing the lift by this spread gives z, which counts how many standard errors away from zero we landed.

## From z to a p-value

For large groups, z behaves like a draw from a standard normal distribution when the null hypothesis is true. The p-value is the probability of drawing something at least as extreme as what we saw, in either direction, because we did not decide in advance which page would be better. A z of about 3 corresponds to a p-value of about 0.0027, which says that a gap this large would appear by chance in fewer than three out of a thousand identical experiments. When the p-value is below the chosen alpha, usually 0.05, the result is called significant.

The math module gives the two-sided p-value directly through the complementary error function, since the tail probability of a standard normal beyond a positive z equals half of erfc of z over the square root of two.

## The interval uses a different standard error

The interval describes the range of lifts that are compatible with the data, without assuming the two rates are equal. It therefore uses each group's own rate for its own variance,

$$
\mathrm{SE}_{unpooled} = \sqrt{\frac{p_c (1 - p_c)}{n_c} + \frac{p_t (1 - p_t)}{n_t}}
$$

and stretches by the normal quantile for the chosen confidence, about 1.96 for 95 percent. An interval that excludes zero agrees with a significant test in almost every case, and its width tells the team how precisely the lift is known. A lift of one point with an interval from 0.35 to 1.65 is a confident win. A lift of one point with an interval from minus two to plus four is a shrug.

## Degenerate experiments

If nobody converted in either group, or everybody did, the pooled rate is zero or one and the standard error collapses to zero. Division is impossible and the honest report is that there is no evidence of a difference, so z is zero and the p-value is one. Production code has to handle this because experiments on rare events sometimes finish a day with zero conversions.

## How it comes up in interviews

An interviewer may ask how you would decide whether a new recommendation model is better than the current one once both can be put in front of users. A newcomer's answer runs like this. I would split users at random into two groups of equal size, show each group one model, and count conversions for a fixed period such as two weeks. Then I would compute the lift and a two-proportion z-test. I would report the p-value together with a 95 percent confidence interval for the lift, so the team sees whether the gap is real and how large it could be.

With about 20,000 users per group a lift of one percentage point on a twelve percent base rate is detectable, and a smaller lift would need more users or more time.

A follow up often asks about checking the result every day and stopping at the first significant reading. Doing so raises the false positive rate well above alpha, so the stopping rule has to be fixed before the experiment starts.