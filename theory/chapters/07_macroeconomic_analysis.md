# Macroeconomic Analysis and Top-Down Portfolio Positioning

Economic fundamentals drive equity returns through multiple channels — discount rates, cash flow expectations, credit conditions, and risk appetite. The preceding chapters construct portfolios from the bottom up: factor scores identify attractive stocks, moment estimation captures return dynamics, and optimization allocates capital across the selected universe. This chapter complements that framework with top-down macroeconomic analysis that translates broad economic conditions into sector rotation decisions, quantitative indicator thresholds, and geographic allocation rules. The treatment is deliberately complementary to the regime-conditional factor tilts presented in the stock pre-selection chapter: where that framework adjusts factor group weights within the composite scoring function, the material here addresses sector-level positioning, macro indicator monitoring, and cross-regional allocation that sit above and around the factor pipeline.

## Business Cycle and Sector Rotation

### The Four-Phase Framework

The business cycle framework divides economic evolution into four phases with distinct sector performance patterns. Dallas Fed research (2019) identifies three macro factors explaining approximately 27 percent of the common variation in equity returns: economic growth (favoring Technology, Consumer Discretionary, Financials), inflation (benefiting Energy, Financials, Materials while hurting Technology), and commodity prices (directly impacting Energy). These macro drivers manifest differently across cycle phases, providing institutional investors with disciplined rotation strategies.

The four phases and their characteristics are:

| Phase       | Growth              | Duration                | Avg. Return            | Monetary Policy     | Yield Curve              |
| ----------- | ------------------- | ----------------------- | ---------------------- | ------------------- | ------------------------ |
| Early cycle | Sharp acceleration  | approximately 1 year    | $>$20% ann.            | Easy, accommodative | Steep                    |
| Mid cycle   | Moderate, sustained | approximately 3 years   | approximately 14% ann. | Gradual tightening  | Normal positive          |
| Late cycle  | Decelerating        | approximately 1.5 years | approximately 5% ann.  | Tight               | Flattening/inverting     |
| Recession   | Negative            | $<$1 year               | Negative               | Easing              | Inverted then steepening |

These phases map onto, but are not identical to, the four-regime classification used for factor tilts in the pre-selection chapter (expansion, slowdown, recession, recovery). The factor tilt regimes are defined by GDP growth direction and unemployment dynamics for the purpose of adjusting composite scoring weights. The business cycle phases here are defined by a broader set of indicators — monetary policy stance, credit conditions, yield curve shape — and target sector allocation rather than factor weights. In practice, early cycle aligns with recovery, mid cycle with expansion, late cycle with slowdown, and recession maps directly. The distinction matters because factor tilts and sector rotation can be applied simultaneously as complementary layers of portfolio positioning.

### Phase-Specific Sector Performance

**Early-cycle phase** (recovery from recession) features sharp growth acceleration, improving credit conditions, and steep yield curves. Top-performing sectors are Consumer Discretionary (100 percent hit rate of outperformance since 1962), Industrials, Real Estate, Financials, Information Technology, and Materials. Defensive sectors — Utilities, Healthcare, Consumer Staples — systematically underperform as investors rotate toward cyclical growth. Entry signals include ISM PMI crossing above 50, declining unemployment claims, narrowing credit spreads, and steepening yield curves.

**Mid-cycle phase** (sustained expansion) represents the longest period with moderate growth, strong credit expansion, and average returns around 14 percent annualized. This phase shows the least sector differentiation — security selection matters more than sector bets. Information Technology and Communication Services demonstrate modest outperformance, while Materials and Utilities slightly lag. The key insight: minimize concentrated sector tilts during mid-cycle and focus on bottom-up stock selection and factor exposures. This is where the factor scoring framework from the pre-selection chapter carries the most weight relative to top-down positioning.

**Late-cycle phase** (peak and slowdown) features decelerating growth, rising inflation, tightening monetary policy, and flattening or inverted yield curves. Energy consistently outperforms as inflation accelerates, while defensive sectors (Consumer Staples, Utilities) begin relative strength. Technology, Consumer Discretionary, and Industrials underperform as growth expectations fall. Critical warning signals include ISM PMI declining below 50, yield curve inversion (2s10s spread turning negative), widening credit spreads, and late-stage Fed tightening cycles.

**Recession phase** (contraction) brings negative GDP growth, declining profits, credit scarcity, and accommodative policy responses. Defensive sectors dominate: Consumer Staples (perfect track record of outperformance), Utilities, and Healthcare provide downside protection through non-cyclical demand. Cyclical sectors — Financials, Industrials, Technology, Real Estate, Consumer Discretionary — systematically underperform. The rotation to defensives should occur before recession onset; yield curve inversion has historically provided 6 to 24 months of advance warning, though the 2022--2024 episode demonstrates that inversion is a necessary but not sufficient condition for recession.

The following table summarizes sector positioning by cycle phase:

| Sector                 | Early Cycle | Mid Cycle | Late Cycle  | Recession   |
| ---------------------- | ----------- | --------- | ----------- | ----------- |
| Consumer Discretionary | Overweight  | Neutral   | Underweight | Underweight |
| Financials             | Overweight  | Neutral   | Underweight | Underweight |
| Industrials            | Overweight  | Neutral   | Underweight | Underweight |
| Materials              | Overweight  | Neutral   | Slight OW   | Underweight |
| Technology             | Overweight  | Slight OW | Underweight | Underweight |
| Energy                 | Neutral     | Neutral   | Overweight  | Neutral     |
| Communication Services | Neutral     | Slight OW | Neutral     | Neutral     |
| Real Estate            | Overweight  | Neutral   | Underweight | Underweight |
| Consumer Staples       | Underweight | Neutral   | Overweight  | Overweight  |
| Healthcare             | Underweight | Neutral   | Neutral     | Overweight  |
| Utilities              | Underweight | Neutral   | Overweight  | Overweight  |

### GICS Classification by Economic Sensitivity

The Global Industry Classification Standard (GICS) provides the institutional framework for sector analysis, dividing equities into 11 sectors, 25 industry groups, 74 industries, and 163 sub-industries. For macroeconomic positioning, the 11 sectors are grouped into three sensitivity categories that determine their response to business cycle dynamics.

**Cyclical sectors (6)** demonstrate high economic sensitivity and typically outperform during expansions:

- **Consumer Discretionary** — highly correlated with consumer confidence and employment
- **Financials** — sensitive to interest rate slopes, credit cycles, and regulatory environments
- **Industrials** — covers manufacturing, transportation, and business services; leading indicators of economic activity
- **Materials** — commodity sensitive and early-cycle oriented
- **Technology** — mixed cyclicality; hardware and semiconductors are cyclical while software demonstrates more stable growth
- **Real Estate** — interest rate sensitive with early-cycle recovery characteristics

**Defensive sectors (3)** provide downside protection through inelastic demand:

- **Consumer Staples** — recession-resistant with stable cash flows
- **Healthcare** — aging demographics provide structural growth independent of cycle
- **Utilities** — regulated monopolies with predictable earnings; bond proxies sensitive to interest rates but defensive during equity stress

**Mixed classification (2)** applies to sectors with heterogeneous sub-industry dynamics:

- **Energy** — highly commodity-dependent with strong inflation correlation; benefits in late cycle but vulnerable to demand destruction in recession
- **Communication Services** — reconstituted in 2018, combining telecoms (defensive) with media and internet (growth-oriented)

For actively managed long-only equity mandates benchmarked against broad indices, institutional sector positioning typically allows $\pm$3 to 5 percentage points of active weight versus benchmark at the sector level, and $\pm$2 to 3 percentage points at the industry group level. These constraints prevent excessive concentration while permitting meaningful active bets. Maximum tracking error budgets of 2 to 4 percent limit aggregate sector tilts. These ranges represent common practice rather than universal standards and vary by mandate type.

## Quantitative Indicator Framework

Translating qualitative macro views into portfolio actions requires specific thresholds and decision frameworks. Three indicators — the ISM Manufacturing PMI, the yield curve slope, and credit spreads — provide complementary signals with measurable decision rules.

### ISM Manufacturing PMI

The ISM Manufacturing PMI serves as the premier real-time economic indicator. Readings above 50 signal expansion, below 50 indicate contraction, and historical analysis shows that PMI above 42.3 correlates with positive overall GDP growth. The indicator is released on the first business day of each month (second business day in January) and is never revised, making it one of the timeliest and most stable macro signals available.

Decision rules translate PMI readings into sector positioning:

| PMI Range | Signal               | Sector Positioning                                        |
| --------- | -------------------- | --------------------------------------------------------- |
| $>$ 52    | Strong expansion     | Cyclical overweights (Industrials, Materials, Technology) |
| 48-52     | Neutral / transition | Maintain neutral positioning                              |
| $<$ 48    | Contraction          | Rotate to defensives (Staples, Healthcare, Utilities)     |
| $<$ 45    | Deep contraction     | Maximum defensive positioning                             |

The PMI's predictive power extends beyond its level to its direction and rate of change. A PMI reading of 51 that has declined from 55 over three months carries different information than a reading of 51 rising from 47. Incorporating the first difference $\Delta \text{PMI}_t = \text{PMI}_t - \text{PMI}_{t-1}$ into the decision framework captures momentum in economic conditions. Persistent negative $\Delta \text{PMI}$ below $-1$ per month signals accelerating deterioration even when the level remains above 50.

### Yield Curve Slope

The yield curve slope, measured as the spread between 10-year and 2-year Treasury yields (the 2s10s spread), provides powerful recession forecasting capability. Every US recession since 1969 has been preceded by a yield curve inversion, with lead times of 6 to 24 months. However, the indicator is not infallible: the 2022--2024 inversion — the longest on record — did not produce a recession, representing either a delayed signal or a false positive that has prompted reassessment of the indicator's unconditional reliability.

Decision rules based on the 2s10s spread:

| 2s10s Spread         | Signal            | Portfolio Implications                                          |
| -------------------- | ----------------- | --------------------------------------------------------------- |
| $>$ +100 bps         | Normal, steep     | Supports Financials and cyclical sectors                        |
| +50 to +100 bps      | Normal            | Neutral positioning                                             |
| 0 to +50 bps         | Flat              | Initiate defensive rotation; reduce Financial exposure          |
| $<$ 0 bps (inverted) | Recession warning | Aggressive defensive positioning; maximum underweight cyclicals |

The yield curve transmits information through two channels. The expectations channel reflects market pricing of future short-term rates: an inverted curve implies that the market expects rate cuts, which historically accompany recessions. The term premium channel captures compensation for duration risk: compressed or negative term premia reduce bank net interest margins and tighten financial conditions, directly impacting the profitability of the Financial sector. Both channels reinforce the same positioning signal — inversion demands defensive rotation.

### Credit Spread Dynamics

Credit spreads offer leading indicators of equity stress, with academic research (Gilchrist and Zakrajšek, 2012) documenting predictive power at horizons of one to four quarters. Lead times vary significantly by episode — from near-simultaneous movement in fast-moving crises (March 2020) to multi-quarter leads in slow-building cycles (2007--2008). Both high-yield (HY) and investment-grade (IG) spreads provide actionable thresholds.

**High-yield spreads** over Treasuries signal risk appetite across a spectrum of conditions. Note that the long-run average HY OAS since 1997 is approximately 540 bps, and the thresholds below are heuristic categories calibrated to institutional practice rather than formally standardized levels:

| HY Spread    | Signal   | Positioning                                                          |
| ------------ | -------- | -------------------------------------------------------------------- |
| $<$ 350 bps  | Risk-on  | Favors cyclicals; maintain equity overweight                         |
| 350-500 bps  | Neutral  | Balanced positioning                                                 |
| 500-800 bps  | Risk-off | Defensive rotation; reduce equity beta                               |
| $>$ 800 bps  | Stressed | Maximum quality bias and defensive allocation                        |
| $>$ 1000 bps | Crisis   | Full defensive; historical crises (GFC: 2,182 bps; COVID: 1,087 bps) |

**Investment-grade spreads** show similar patterns at tighter levels. Extremely tight IG spreads (below 80 bps, approaching historical minimums) can paradoxically signal late-cycle complacency, as tight spreads often precede widening episodes. This non-monotonic relationship requires monitoring both the level and the direction of spread changes.

The rate of change in credit spreads matters as much as the level. Rapid widening of 50 bps or more over a one-month period warrants immediate defensive action regardless of the absolute spread level, as it signals a regime shift in credit market risk appetite that typically precedes equity market weakness.

### Integrated Multi-Indicator Decision Framework

No single indicator is sufficient for robust macro positioning. The integrated framework combines all three indicators into a composite signal with explicit decision rules:

**Regime 1 — Expansionary (all signals aligned bullish)**: When ISM $>$ 50, 2s10s spread $>$ +100 bps, and HY spreads $<$ 400 bps, maintain aggressive cyclical overweights. All three indicators confirm economic strength, supporting risk-on positioning across sectors and factors.

**Regime 2 — Transitional (mixed signals)**: When ISM approaches 50, yield curve flattens below 50 bps, and credit spreads widen above 450 bps, initiate defensive rotation. The combination of deteriorating manufacturing activity with tightening financial conditions signals late-cycle dynamics. Begin reducing cyclical exposure and increasing defensive allocations.

**Regime 3 — Contractionary (all signals aligned bearish)**: When ISM falls below 48, yield curve inverts, and HY spreads exceed 500 bps, implement maximum defensive positioning. The confluence of manufacturing contraction, inverted term structure, and elevated credit risk demands full rotation to Consumer Staples, Healthcare, and Utilities with minimal cyclical exposure.

The framework assigns equal weight to each indicator for regime classification. A simple scoring system maps each indicator to $\{-1, 0, +1\}$ based on its threshold breakpoints, and the sum determines the composite signal:

$$
S_t = s_t^{\text{PMI}} + s_t^{\text{2s10s}} + s_t^{\text{HY}}
$$

where each component score is:

$$
s_t^{\text{PMI}} = \begin{cases} +1 & \text{if PMI}_t > 52 \\ 0 & \text{if } 48 \leq \text{PMI}_t \leq 52 \\ -1 & \text{if PMI}_t < 48 \end{cases}
$$

$$
s_t^{\text{2s10s}} = \begin{cases} +1 & \text{if spread}_t > 100 \text{ bps} \\ 0 & \text{if } 0 \leq \text{spread}_t \leq 100 \text{ bps} \\ -1 & \text{if spread}_t < 0 \text{ bps} \end{cases}
$$

$$
s_t^{\text{HY}} = \begin{cases} +1 & \text{if spread}_t < 350 \text{ bps} \\ 0 & \text{if } 350 \leq \text{spread}_t \leq 500 \text{ bps} \\ -1 & \text{if spread}_t > 500 \text{ bps (entering risk-off territory)} \end{cases}
$$

A composite score of $S_t \geq 2$ maps to the expansionary regime, $-1 \leq S_t \leq 1$ to the transitional regime, and $S_t \leq -2$ to the contractionary regime. This scoring approach is deliberately simple; more sophisticated models (logistic regression, random forests) can improve classification accuracy but add complexity that may not be justified given the inherent imprecision of macro forecasting.

![Fig. 17: Integrated Macro Indicator Dashboard](../figures/ch07/fig_17_indicator_dashboard.png)
_Figure 17: Time series of the three macro indicators with decision thresholds overlaid. Shaded regions indicate the composite regime classification. The integrated framework captures regime transitions earlier than any single indicator._

## Geographic Allocation

### Developed Versus Emerging Markets

Global equity allocation depends on growth differentials, currency dynamics, monetary policy divergence, and regional valuations. Emerging markets (EM) deliver higher growth — historically 2 to 3 percentage points above developed markets (DM) — but with greater volatility, political risk, and currency sensitivity. The allocation decision reduces to whether the EM growth premium adequately compensates for these additional risks.

Decision rules for DM/EM allocation:

$$
\text{EM tilt} = \begin{cases} \text{Overweight} & \text{if } \Delta g_{\text{EM-DM}} > 200 \text{ bps and USD weakening} \\ \text{Neutral} & \text{if } 100 < \Delta g_{\text{EM-DM}} \leq 200 \text{ bps} \\ \text{Underweight} & \text{if } \Delta g_{\text{EM-DM}} \leq 100 \text{ bps or USD strengthening}
\end{cases}
$$

where $\Delta g_{\text{EM-DM}}$ is the projected GDP growth differential between emerging and developed markets. When the Fed tightens and the USD strengthens, capital flows favor developed markets; when EM growth premiums are wide and the dollar cycle turns, EM allocations become attractive.

Regional allocation within these broad categories follows a framework that considers structural growth drivers, valuation, monetary policy stance, and geopolitical risk:

| Region        | Key Drivers                                                   | Cyclical Sensitivity         |
| ------------- | ------------------------------------------------------------- | ---------------------------- |
| United States | Technology leadership, profit margins, AI infrastructure      | Moderate; tech-driven        |
| Europe        | Fiscal stimulus, defense spending, structural challenges      | High; export-dependent       |
| Japan         | Corporate governance reform, BOJ normalization                | Moderate; currency-sensitive |
| China         | Stimulus vs. structural headwinds (demographics, real estate) | High; policy-dependent       |
| India         | Demographics, manufacturing shift, sustained growth           | Moderate; domestic-driven    |
| Other EM      | Monetary policy, commodity exposure, supply chain shifts      | High; commodity/currency     |

### Currency Hedging Decisions

Currency hedging decisions significantly impact returns for international allocations. The hedging framework follows institutional best practice:

**Developed market exposure** is hedged 50 to 100 percent of currency risk, particularly when the USD is expected to strengthen. Hedging costs for G10 currencies are typically modest (reflecting interest rate differentials), and removing currency volatility from developed market positions isolates the equity return signal.

**Emerging market exposure** is left largely unhedged for two reasons. First, EM hedging costs are substantially higher, reflecting wide interest rate differentials that erode returns. Second, EM currency exposure provides diversification benefits — EM currencies tend to appreciate during global risk-on environments when EM equities also perform well, amplifying returns, and depreciate during risk-off episodes, partially offsetting equity losses through reduced portfolio weight in local currency terms.

Dynamic hedging based on currency valuations and interest rate differentials can add 50 to 150 basis points annually. The decision to hedge is itself a macro view: hedging USD/EUR when US-Europe rate differentials are wide locks in positive carry, while removing hedges when rate differentials narrow avoids paying away the carry cost.

## Integration with the Existing Pipeline

### Connection to Regime Classification

The macro indicators described above feed directly into the regime classification function used by the factor pipeline. The existing `classify_regime()` function in the factors module uses GDP growth direction and yield-spread heuristics to assign one of four regimes (expansion, slowdown, recession, recovery). The quantitative indicator framework presented here provides the raw inputs for that classification:

- ISM PMI level and direction map to GDP growth assessment
- The 2s10s yield curve spread is directly used as the yield-spread input
- Credit spread dynamics provide supplementary confirmation

The composite score $S_t$ from the integrated framework can serve as an alternative or complementary input to regime classification. When $S_t \geq 2$, the regime maps to expansion or recovery; when $-1 \leq S_t \leq 1$, the regime maps to expansion (if PMI rising) or slowdown (if PMI falling); when $S_t \leq -2$, the regime maps to recession.

### Connection to Factor Tilts

Once the macro regime is identified, it drives the multiplicative factor group tilts described in the stock pre-selection chapter. The regime-aware scoring formula applies tilts to baseline group weights:

$$
w_g^{\text{regime}} = w_g^{\text{baseline}} \times m_g(R_t)
$$

The macro indicators determine $R_t$, and the tilt multipliers $m_g(R_t)$ adjust factor weights accordingly — overweighting value and momentum during expansions and recoveries, overweighting quality and low volatility during slowdowns and recessions. The sector rotation layer described in this chapter operates in parallel: factor tilts adjust which stocks score highest within the selected universe, while sector positioning adjusts the allocation across sectors in the optimized portfolio.

### Connection to Moment Estimation

Macro regime information also feeds into moment estimation through the HMM-blended estimators described in the moment estimation chapter. The `HMMBlendedMu` and `HMMBlendedCovariance` estimators use regime-probability-weighted blending of per-regime moments. Macro indicators can inform the regime probability vector used for blending:

$$
\boldsymbol{\mu}_{\text{blended}} = \sum_{k=1}^{K} \pi_k(S_t) \, \boldsymbol{\mu}_k, \quad \boldsymbol{\Sigma}_{\text{blended}} = \sum_{k=1}^{K} \pi_k(S_t) \left[\boldsymbol{\Sigma}_k + (\boldsymbol{\mu}_k - \boldsymbol{\mu}_{\text{blended}})(\boldsymbol{\mu}_k - \boldsymbol{\mu}_{\text{blended}})^\top\right]
$$

where $\pi_k(S_t)$ are regime probabilities conditioned on the macro composite score. When the integrated framework signals a clear regime ($|S_t| \geq 2$), the corresponding regime receives elevated probability weight; when signals are mixed ($|S_t| \leq 1$), probabilities remain closer to the HMM's statistical estimates.

### Connection to View Integration

Macro analysis provides a natural source of views for the Black-Litterman framework described in the view integration chapter. Sector rotation signals translate directly into relative views:

- During early cycle with $S_t \geq 2$: express views that Consumer Discretionary and Industrials outperform Consumer Staples and Utilities by the historical average differential (approximately 5 to 8 percent annualized)
- During late cycle with $S_t \leq -1$: express views that defensive sectors outperform cyclicals

Geographic allocation signals similarly translate into views on country or regional ETF exposures. The view confidence parameter $\boldsymbol{\Omega}$ should reflect the clarity of the macro signal: high-conviction views (when all three indicators are aligned) warrant tighter uncertainty, while mixed-signal environments call for wider uncertainty that keeps the posterior closer to the equilibrium prior.

### Detection Lag and Practical Limitations

Real-time regime detection lags the true economic regime by one to three months due to publication delays in macroeconomic data and the need for confirmation across multiple indicators. ISM PMI is the timeliest indicator (released on the first business day of the following month), while GDP data and employment statistics arrive with longer delays and are subject to revision. This lag means that by the time a regime transition is confirmed, markets may have already partially priced it in.

Asness and co-authors argue skeptically that much of the theoretical benefit of macro timing is eroded once detection lags and transaction costs are incorporated (Asness, Ilmanen, and Maloney, 2015; Asness, Chandra, Ilmanen, and Israel, 2017). The practical implication is that macro positioning should be implemented as moderate tilts around a diversified baseline rather than aggressive binary bets. The sector overweights and underweights in the cycle phase table above are constrained to institutional norms ($\pm$3 to 5 percentage points) precisely because detection uncertainty makes larger tilts unreliable. This is consistent with the treatment of regime-conditional factor tilts in the pre-selection chapter, where tilts are described as marginal adjustments on top of robust static factors.

## News Sentiment as a Complementary Macro Signal

The quantitative indicators described above — PMI, yield curve, credit spreads — are derived from structured economic data released on fixed schedules. News flow provides a complementary channel that captures shifts in growth expectations, policy uncertainty, and market risk appetite with near-zero publication lag. Country-level news sentiment does not replace hard data indicators but augments them by filling the gap between data releases, when qualitative information accumulates in financial media before it materialises in official statistics. The daily country-level news summary pipeline operationalises this channel by producing structured sentiment assessments from recent financial coverage.

### Motivation and Theoretical Grounding

Tetlock (2007) established the empirical foundation for text-based financial signals by demonstrating that media pessimism — measured by the fraction of negative words in Wall Street Journal columns — predicts near-term stock market declines and elevated trading volume. The effect is economically significant at daily and weekly horizons, though it partially reverses over longer periods, suggesting that sentiment captures temporary mispricing driven by investor attention rather than permanent changes in fundamentals.

Loughran and McDonald (2011) refined word-list construction by showing that standard Harvard General Inquirer negative words are poor proxies for financial negativity. Terms like "liability", "tax", and "capital" register as negative in the Harvard list but carry neutral or positive connotation in financial contexts. Their domain-specific financial word list substantially improves classification accuracy for SEC filings and earnings releases, establishing that sentiment measurement requires lexicons calibrated to the target domain rather than general-purpose dictionaries.

Baker, Bloom, and Davis (2016) extended the approach to macroeconomic uncertainty measurement through their **Economic Policy Uncertainty** (EPU) index, which combines newspaper coverage of economic and policy uncertainty with tax code expiration data and forecaster disagreement. High EPU correlates with reduced corporate investment, lower employment growth, and suppressed output, validating that aggregate news content carries macro-level information beyond individual asset returns. The EPU framework captures uncertainty rather than directional sentiment, making it complementary to the polarity-based approach described below.

Collectively, these findings establish that country-level news flow contains time-varying signals about growth expectations, policy uncertainty, and risk appetite that complement the hard data captured by PMI and credit spreads — and that are available with lower publication lag than any of the three quantitative indicators in the existing framework.

### The News Sentiment Score

The **news sentiment score** $\psi_{c,t} \in [-1, 1]$ is a scalar produced for country $c$ on date $t$ that summarises the directional tone of recent financial coverage. Its conceptual foundation is the Loughran-McDonald polarity ratio for a single document $d$:

$$
\text{Pol}(d) = \frac{N_{\text{pos}}(d) - N_{\text{neg}}(d)}{N_{\text{pos}}(d) + N_{\text{neg}}(d) + 1}
$$

where $N_{\text{pos}}(d)$ and $N_{\text{neg}}(d)$ count positive and negative terms according to the financial-domain lexicon. The denominator includes a smoothing constant of 1 to handle documents with no sentiment-bearing terms. This formula contextualises what the LLM approximates when producing a continuous sentiment score from article text: the model implicitly performs polarity classification but with richer contextual understanding than a bag-of-words lexicon.

The aggregate country score across $n$ articles available on day $t$ is:

$$
\psi_{c,t} = \frac{1}{n} \sum_{i=1}^{n} \omega_i \cdot \text{Pol}(d_i)
$$

where $\omega_i$ is a conceptual article-relevance weight discussed in the limitations section below. The score $\psi_{c,t}$ maps to the same $[-1, 1]$ interval as the `sentiment_score` field produced by the `SummarizeCountryNews` function in the news pipeline.

To integrate sentiment into the macro decision framework, the composite score $S_t$ from the three-indicator system is extended with a fourth component:

$$
S_t^{\text{augmented}} = s_t^{\text{PMI}} + s_t^{\text{2s10s}} + s_t^{\text{HY}} + s_t^{\text{sent}}
$$

where the sentiment component maps the continuous score to a discrete signal:

$$
s_t^{\text{sent}} = \begin{cases} +1 & \text{if } \psi_{c,t} > 0.3 \\ 0 & \text{if } -0.3 \leq \psi_{c,t} \leq 0.3 \\ -1 & \text{if } \psi_{c,t} < -0.3 \end{cases}
$$

The $\pm 0.3$ thresholds match the calibration boundaries used in the `SummarizeCountryNews` prompt, which classifies sentiment as BULLISH when the score exceeds $+0.3$ and BEARISH when it falls below $-0.3$. This ensures that the discrete component score is consistent with the LLM's own classification. The wide neutral band ($[-0.3, +0.3]$) reflects the inherent noise in text-derived signals and prevents marginal sentiment readings from influencing the composite score.

When incorporating the sentiment component, the regime thresholds should be re-calibrated to maintain equivalent sensitivity to the original three-indicator framework. With four components, $S_t^{\text{augmented}}$ ranges from $-4$ to $+4$ rather than $-3$ to $+3$. Appropriate thresholds are $S_t^{\text{augmented}} \geq 3$ for the expansionary regime and $S_t^{\text{augmented}} \leq -3$ for the contractionary regime, preserving the requirement that at least three of four indicators must be aligned before triggering aggressive positioning.

### Implementation in the Country News Pipeline

The daily country-level news summary pipeline operationalises the sentiment score through a structured LLM workflow. The pipeline fetches articles from the preceding 24 hours, maps them to countries via predefined ticker-to-country and query-to-country mappings, groups the articles by country, and requires a minimum of three articles before invoking the LLM to avoid producing sentiment estimates from insufficient coverage.

The `SummarizeCountryNews` function produces a structured output containing three fields: a 3--5 sentence macro briefing (`summary`) that captures the key economic themes, a discrete sentiment label (`sentiment`) drawn from the set BULLISH, BEARISH, NEUTRAL, and MIXED, and a continuous sentiment score (`sentiment_score`) corresponding to $\psi_{c,t}$. The discrete label maps directly to the component score $s_t^{\text{sent}}$: BULLISH maps to $+1$, BEARISH to $-1$, and both NEUTRAL and MIXED to $0$.

Output is persisted to the `macro_news_summaries` table with cache gating — a country is not re-summarised within the same calendar day unless a forced refresh is requested. This prevents redundant LLM invocations while ensuring that the sentiment signal updates at daily frequency. Countries currently covered by the mapping include the United States, the United Kingdom, Germany, and France. Articles with global or commodity focus that cannot be attributed to a specific country are excluded from country-level sentiment computation to avoid diluting geographic specificity.

The continuous `sentiment_score` is the operationalisation of $\psi_{c,t}$; the discrete `sentiment` label provides a human-readable classification that maps to the $\{+1, 0, -1\}$ component score. Both fields are stored and available to downstream consumers, allowing the integration framework to use either the continuous score for granular blending or the discrete label for threshold-based regime classification.

### Limitations and the Conceptual Status of Lambda

**The article-relevance weight $\omega_i$ as conceptual framing.** The weighting term $\omega_i$ in the polarity formula represents article relevance — a central bank statement should carry more weight than a speculative opinion piece, and a headline from a major financial wire service should outweigh a regional newspaper column. In practice, the pipeline assigns equal weight to all articles ($\omega_i = 1$ for all $i$), with the LLM performing implicit relevance filtering through its prompt guidance to focus on macro-relevant content. The weight $\omega_i$ is therefore a conceptual parameter that identifies where relevance weighting would be inserted if the pipeline were extended to incorporate source quality, article prominence, or topic relevance scoring. It is not a tunable hyperparameter in the current implementation, and there is no calibration procedure for it. This distinction is important: the formula presents the general framework, but the implementation makes a specific simplifying choice that may be revisited as the pipeline matures.

**Detection lag and signal noise.** News articles appear with near-zero publication lag relative to PMI (released monthly with a one-month delay) or GDP (released quarterly with multi-month delays and subsequent revisions). This timeliness advantage is partially offset by signal noise: any individual article may reflect editorial stance, political framing, or sensationalism rather than economic fundamentals, and the LLM summary condenses heterogeneous coverage into a single scalar. The sentiment score should be treated as carrying wide uncertainty — comparable to a mixed-signal macro environment — until corroborated by movements in the hard data indicators.

**Coverage asymmetry.** The current implementation covers four countries (USA, UK, Germany, France), while the quantitative indicator framework covers only the United States by default in the calibration service. Cross-country consistency of sentiment scores is not guaranteed because the LLM prompt does not normalise for country-level differences in press freedom, media concentration, reporting conventions, or the relative volume of English-language coverage. Loughran and McDonald (2011) note that even lexicon-based approaches require domain-specific calibration; the LLM approach inherits a similar calibration dependency, amplified by the additional dimension of cross-country variation.

**Mean reversion in sentiment-driven effects.** Tetlock (2007) documents partial reversal of sentiment-driven price effects over longer horizons: high media pessimism predicts returns over the following trading week but not over the following quarter. This mean-reversion pattern implies that sentiment signals should carry lower weight in the composite score when the investment horizon extends beyond a few weeks. The wide neutral band ($\pm 0.3$) in the discrete component score partially addresses this by classifying moderate sentiment readings as zero contribution, but users of the augmented composite score should be aware that the sentiment component is most informative at short horizons and least informative for strategic allocation decisions.

**Relationship to EPU.** Baker, Bloom, and Davis (2016) construct the EPU index using newspaper search counts rather than polarity scoring, and it captures uncertainty rather than directional sentiment. The `sentiment_score` produced by the pipeline is closer in spirit to Tetlock (2007) and Loughran and McDonald (2011) — measuring directional polarity — than to the EPU framework. The two signals are complementary: a country with high EPU and positive directional sentiment presents a mixed picture — positive growth expectations alongside elevated policy uncertainty — which maps naturally to the MIXED classification label. A complete macro sentiment framework would ideally incorporate both directional polarity and uncertainty measurement, though the current pipeline focuses on the former.

### Intraday Refresh and Retention

The daily summarization described above captures the end-of-day snapshot of country-level sentiment. Between daily runs, news flow continues to arrive and may shift the sentiment picture — particularly around central bank announcements, employment reports, or geopolitical events that break during trading hours. A 30-minute incremental scheduler addresses this gap by re-processing only countries that have received new articles since the previous tick.

On each tick the scheduler queries the `macro_news` table for articles with `publish_time` after the previous tick's timestamp. Articles are mapped to countries through the same ticker-to-country and query-to-country dictionaries used by the daily pipeline. Only countries with at least one new article are re-summarized with `force_refresh=True`, bypassing the cache gate and invoking the LLM. Countries with no new articles retain their existing summary without additional LLM cost. This incremental design keeps LLM invocations proportional to actual news activity rather than running a fixed number of calls per tick.

The scheduler also enforces a 90-day retention policy on the `macro_news_summaries` table. On every tick, rows with `summary_date` older than 90 days are deleted. This prevents unbounded table growth while preserving sufficient history for trend analysis and backtesting of the sentiment signal's predictive value.

Error isolation operates at two levels. Per-country errors within `generate_country_summaries` are caught and logged without blocking other countries. The scheduler's own tick function is wrapped in a top-level exception handler so that a failure in one tick — whether from a database timeout, LLM API error, or transient network issue — does not terminate the daemon thread. The scheduler simply logs the error and waits for the next interval.

\newpage
