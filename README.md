**IPL Win Probability Predictor**

A ball-by-ball machine learning model that estimates the chasing team's chance of winning from any state of an IPL run chase the same class of model broadcasters run live on screen.


Live app: <!-- paste your Streamlit URL here --> · Built with Python · scikit-learn · Streamlit




**The question**

At any point in a run chase say 62 needed off 40 balls with 5 wickets in hand what is the batting team's real probability of getting over the line? Fans, commentators and even captains rely on gut feel anchored to a handful of famous chases. This project replaces that gut feel with a number grounded in every IPL chase ever recorded.

**Key finding**

Chases that need ~120 off the last 10 overs succeed just 22% of the time. Human intuition, biased by memorable run-fests, badly overestimates it. Quantifying that gap is the whole point of the app.

**Results**

1. Metric Score What it means AUC0.865 Strong separation of wins vs losses on an honest
2. unseen test set Brier score 0.151 Probabilities are well-calibrated
3. not just directionally correct Training states 138,000+ Ball-by-ball chase snapshots from 1,100+ IPL matches

**Why this project is honest about leakage**

An early Random Forest scored 99% accuracy a classic red flag. The cause was data leakage: rows from the same match appearing in both train and test sets, letting the model "memorise" outcomes.

**The fix defined the credibility of the whole project:**
1. Match-level train/test splits:  an entire match is either training data or test data, never both. The model is judged only on matches it has never seen.
2. Model choice : a calibrated Logistic Regression was preferred over a flashier tree ensemble because its probabilities are trustworthy and its behaviour is interpretable.
3. Reported numbers are the post-leakage, honest ones (AUC 0.865), not the inflated 99%.


**How it works**

1. Data — raw Cricsheet ball-by-ball JSON for IPL matches.
2. Feature engineering — from each delivery in the second innings, derive the match state: runs required, balls remaining, wickets in hand, current & required run rate, target, etc.
3. Label — did the chasing team ultimately win? (1/0)
4. Model — Logistic Regression with probability calibration, validated with match-level splits.
5. App — a Streamlit interface where you enter a live match state and get an instant win-probability read-out.


**Tech stack**
- Python · pandas · scikit-learn · Logistic Regression · probability calibration · Streamlit

