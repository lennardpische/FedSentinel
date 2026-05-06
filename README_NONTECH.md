# FedSentinel — Plain English Guide

---

## What is this project?

Eight times a year, a group of people called the **Federal Open Market Committee (FOMC)** meets in Washington D.C. and decides what to do with interest rates in the United States. After every meeting, they release a short statement — usually just a few paragraphs — describing how they see the economy and what they decided.

These statements are obsessively read by traders, economists, and analysts all over the world. Why? Because interest rates affect everything — mortgage rates, stock prices, the cost of business loans, the strength of the dollar. Even a single word change ("patient" becoming "data-dependent") can move markets by billions of dollars.

**FedSentinel is an AI system that reads these statements and tries to predict what the Fed will decide at the *next* meeting.** Will they cut rates? Hold them steady? Raise them?

---

## Why is this hard?

A few reasons:

**1. The language is deliberately vague.** The Fed doesn't want to commit to anything too explicitly, so the statements are carefully worded to signal direction without making promises. You have to read between the lines.

**2. There isn't much data.** The Fed only meets 8 times a year. Even going back to 2003, that's only about 170 usable statements. In machine learning, that's a very small dataset. Most AI models are trained on millions of examples — we have 170.

**3. The world changes.** A statement from 2010 (when rates were near zero after the financial crisis) looks very different from one in 2022 (when inflation was surging). The model has to handle very different economic regimes.

---

## Step 1: Collecting the data

The first thing FedSentinel does is download every FOMC statement from the Federal Reserve's website, going back to 2003. This is done automatically by a **web scraper** — a program that visits web pages and saves their text, the same way you might copy-paste an article, except it does it for 194 documents automatically.

The Fed's website has changed its structure several times over the decades, so the scraper had to be taught to handle three different URL and page formats depending on the year. Each statement gets saved as a text file named by its date (e.g. `20231213_Statement.txt`).

---

## Step 2: Labeling the data

Before we can train a model, we need to tell it what the right answer was for each statement. We do this automatically by searching each statement for phrases like:

- *"decided to raise the target range"* → **HIKE**
- *"decided to lower the target range"* → **CUT**
- *"decided to maintain the target range"* → **HOLD**

But here's the key twist: the label we attach to each statement is the decision from the ***next*** meeting, not the current one. Why? Because we want to predict the future. So the model is trained to read statement #47 and predict what decision will be announced at meeting #48.

---

## Step 3: Reading the statements with AI

Raw text is just letters and words — a computer can't do math on it directly. We need to convert each statement into numbers first.

We use a model called **FinBERT** to do this. FinBERT is an AI that was trained on millions of financial documents — news articles, earnings reports, analyst notes — so it already "understands" financial language better than a generic model would.

FinBERT reads each statement and converts it into a list of **768 numbers**. This list is called an **embedding**, and it captures the meaning of the text in a way a computer can work with. Two statements that sound similar will have similar embeddings; two that sound very different will have very different ones.

**One important trick:** before we feed each statement to FinBERT, we hide the sentence that announces the actual rate decision (e.g. *"The Committee decided to raise..."*). If we left it in, the model would just memorize "raise = HIKE" instead of learning anything about the economic reasoning in the rest of the text. We want it to learn from the language about inflation, employment, and growth — not from the answer being handed to it.

### What is mean pooling?

FinBERT reads a statement word by word and produces a number-representation for each word in context. To get a single representation for the whole statement, we take the **average** of all the word representations. This is called "mean pooling." It gives a more complete picture of the document than just using the first word's representation.

---

## Step 4: Adding more context

The raw FinBERT embedding is good, but we can do better by adding more information alongside it. For each meeting, we build a richer picture using:

**Previous meeting's embedding**
The model also gets the embedding from the previous statement. This lets it detect shifts in tone — if last month's statement was very hawkish (inflation-focused) and this one suddenly softens, that's a meaningful signal.

**Sentiment drift**
A single number measuring how much the language changed since the last meeting. A score near 0 means the statement is almost identical to the previous one (a HOLD signal). A high score means the language shifted significantly (potential change ahead).

**The current interest rate**
The Fed Funds Rate going into the meeting. This matters a lot — you can't really hike rates from 0%, and you can't cut them from 0% either. The model needs to know where rates are starting from. Critically, we use the rate from *before* the meeting's decision, so we're not accidentally giving the model information it wouldn't have at prediction time.

**Keyword counts**
Simple counts of hawkish words (like "inflation," "restrictive," "elevated") and dovish words (like "employment," "patient," "uncertainty"). These give the model quick explicit signal about the statement's tone.

All of this gets combined into a single list of **1,541 numbers** per meeting.

---

## Step 5: Training the classifier

Now we have 170 meetings, each represented by 1,541 numbers and a label (CUT / HOLD / HIKE). We split them chronologically:

- **65% for training** — the model learns from these
- **15% for validation** — used to tune settings without peeking at the test set
- **20% for testing** — held out completely until the very end to measure real performance

Two different classifiers are trained and compared:

### The SVM (Support Vector Machine)

Imagine plotting all 170 meetings in space, where each meeting's position is determined by its 1,541 features. The SVM tries to find the best "walls" that divide the CUTs from the HOLDs from the HIKEs, with as much breathing room on each side as possible. It's a well-established technique that works surprisingly well when you don't have much data.

### The Transformer

This is a small neural network that looks at the last **8 meetings in sequence** rather than treating each one independently. The idea is that the Fed's decisions have momentum — a series of HIKEs is more likely to continue or pause than to suddenly reverse. The Transformer can pick up on these multi-meeting patterns.

It only runs when there are at least 150 labeled samples, because it needs enough data to learn sequential patterns without just memorizing the training set.

Whichever model scores higher on the validation set gets used for final evaluation on the test set.

---

## Step 6: Results

| Model | Validation Score |
|-------|-----------------|
| SVM | 0.447 |
| Transformer | 0.447 |
| Logistic Regression | 0.379 |

The SVM and Transformer tied, so the SVM was used (simpler model wins ties). On the test set (2022–2026), the model achieved **50% accuracy** with a macro-F1 of 0.41.

The model does well on HOLD predictions but struggles with HIKEs. This is mostly a data problem: the 2022–2023 hiking cycle was the most aggressive in 40 years, and the model had barely seen any hiking-cycle data during training. It's a bit like training someone to recognize summer and winter weather, then testing them during a hurricane.

---

## What the score means

The metric used is **macro-F1**, which averages performance across all three classes equally. This matters because the data is imbalanced — 71% of meetings were HOLDs, so a model that just always guesses HOLD would get 71% accuracy but be completely useless for detecting cuts and hikes. Macro-F1 penalizes that kind of lazy prediction.

A score of 0.447 out of 1.0 might sound low, but for a 3-class prediction problem on 170 examples covering very different economic regimes, it's meaningful. Random guessing would score around 0.33.

---

## What's next

- **More data from pre-2003** — the Fed's website doesn't have consistent digital records before 2003, but with some effort these could be sourced elsewhere
- **Fix 23 dropped statements** — older statements (2003–2006) used slightly different phrasing that our label detector doesn't catch yet; adding these would push past 200 training samples
- **Paragraph-level features** — instead of embedding the whole statement as one chunk, embed each paragraph separately (inflation section, labor market section, forward guidance) so the model can tell *which part* of the statement is driving the prediction
- **Dashboard** — a simple web interface to visualize predictions, drift over time, and probability breakdowns for each meeting
