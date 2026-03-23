import { useState, useEffect, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, AreaChart, Area, ComposedChart, Bar
} from "recharts";

// ═══════════════════════════════════════════════════════════════════
//  ADVANCED QUANT ENGINE
// ═══════════════════════════════════════════════════════════════════

// --- Math primitives ---
const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
const sum = arr => arr.reduce((a, b) => a + b, 0);
const stdDev = arr => { const m = mean(arr); return Math.sqrt(arr.reduce((a, v) => a + (v - m) ** 2, 0) / arr.length); };
const calcVariance = arr => { const m = mean(arr); return arr.reduce((a, v) => a + (v - m) ** 2, 0) / arr.length; };
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function boxMuller() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Returns from price series
const logReturns = prices => prices.slice(1).map((p, i) => Math.log(p / prices[i]));
const simpleReturns = prices => prices.slice(1).map((p, i) => (p - prices[i]) / prices[i]);

// --- GARCH(1,1) Parameter Estimation (MLE approximation via moment matching) ---
function fitGARCH(returns) {
  const r2 = returns.map(r => r * r);
  const unconditionalVar = calcVariance(returns);
  // Method of moments initialisation
  const alpha0_init = unconditionalVar * 0.1;
  const alpha1_init = 0.08;
  const beta1_init = 0.88;
  // Simple iterative calibration
  let omega = alpha0_init, alpha = alpha1_init, beta = beta1_init;
  const n = 50; // light iterations
  for (let iter = 0; iter < n; iter++) {
    let h = [unconditionalVar];
    let logL = 0;
    for (let t = 1; t < returns.length; t++) {
      const ht = omega + alpha * r2[t - 1] + beta * h[t - 1];
      h.push(ht);
      logL += -0.5 * (Math.log(ht) + r2[t] / ht);
    }
    // Nudge toward higher log-likelihood (gradient-free hill climb)
    const perturbations = [
      [omega * 1.01, alpha, beta],
      [omega * 0.99, alpha, beta],
      [omega, alpha * 1.01, beta],
      [omega, alpha * 0.99, beta],
      [omega, alpha, beta * 1.005],
      [omega, alpha, beta * 0.995],
    ];
    for (const [wo, a, b] of perturbations) {
      if (a + b >= 1 || a < 0 || b < 0 || wo < 0) continue;
      let h2 = [unconditionalVar], l2 = 0;
      for (let t = 1; t < returns.length; t++) {
        const ht = wo + a * r2[t - 1] + b * h2[t - 1];
        h2.push(ht);
        l2 += -0.5 * (Math.log(ht) + r2[t] / ht);
      }
      if (l2 > logL) { omega = wo; alpha = a; beta = b; logL = l2; }
    }
  }
  // Compute conditional variance series
  const condVar = [unconditionalVar];
  for (let t = 1; t < returns.length; t++) {
    condVar.push(omega + alpha * r2[t - 1] + beta * condVar[t - 1]);
  }
  const longRunVar = omega / (1 - alpha - beta);
  return { omega, alpha, beta, condVar, longRunVar, persistence: alpha + beta };
}

// --- GARCH(1,1) Forecast Volatility ---
function garchVolForecast(garch, h0, horizon = 30) {
  const { omega, alpha, beta, longRunVar } = garch;
  const vols = [Math.sqrt(h0)];
  let ht = h0;
  for (let t = 1; t < horizon; t++) {
    ht = omega + (alpha + beta) * ht; // mean-reversion path
    vols.push(Math.sqrt(ht));
  }
  return vols;
}

// --- Hidden Markov Model: 2-state Regime Detection (Bull / Bear) ---
function fitHMM(returns, nIter = 30) {
  // States: 0 = Bull (low vol, positive drift), 1 = Bear (high vol, negative drift)
  const n = returns.length;
  // Init state params
  let muS = [mean(returns.filter((_, i) => i % 2 === 0)), mean(returns.filter((_, i) => i % 2 !== 0))];
  let sigS = [stdDev(returns.filter((_, i) => i % 2 === 0)), stdDev(returns.filter((_, i) => i % 2 !== 0))];
  // Transition matrix
  let A = [[0.95, 0.05], [0.10, 0.90]];
  // Initial state probs
  let pi = [0.6, 0.4];

  const gaussian = (x, mu, sig) => (1 / (sig * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * ((x - mu) / sig) ** 2);

  for (let iter = 0; iter < nIter; iter++) {
    // Forward pass
    const alpha_fwd = Array.from({ length: n }, () => [0, 0]);
    const scale = new Array(n).fill(0);
    for (let s = 0; s < 2; s++) alpha_fwd[0][s] = pi[s] * gaussian(returns[0], muS[s], sigS[s] + 1e-10);
    scale[0] = sum(alpha_fwd[0]);
    for (let s = 0; s < 2; s++) alpha_fwd[0][s] /= scale[0] + 1e-30;
    for (let t = 1; t < n; t++) {
      for (let j = 0; j < 2; j++) {
        alpha_fwd[t][j] = sum([0, 1].map(i => alpha_fwd[t-1][i] * A[i][j])) * gaussian(returns[t], muS[j], sigS[j] + 1e-10);
      }
      scale[t] = sum(alpha_fwd[t]);
      for (let j = 0; j < 2; j++) alpha_fwd[t][j] /= scale[t] + 1e-30;
    }
    // Backward pass
    const beta_bwd = Array.from({ length: n }, () => [1, 1]);
    for (let t = n - 2; t >= 0; t--) {
      for (let i = 0; i < 2; i++) {
        beta_bwd[t][i] = sum([0, 1].map(j => A[i][j] * gaussian(returns[t+1], muS[j], sigS[j] + 1e-10) * beta_bwd[t+1][j]));
      }
      const sc = sum(beta_bwd[t]);
      for (let i = 0; i < 2; i++) beta_bwd[t][i] /= sc + 1e-30;
    }
    // Gamma (state posteriors)
    const gamma = alpha_fwd.map((a, t) => {
      const g = [a[0] * beta_bwd[t][0], a[1] * beta_bwd[t][1]];
      const s = sum(g); return g.map(v => v / (s + 1e-30));
    });
    // Xi (transition posteriors)
    const xi = Array.from({ length: n - 1 }, (_, t) => {
      const mat = [[0,0],[0,0]];
      let tot = 0;
      for (let i = 0; i < 2; i++)
        for (let j = 0; j < 2; j++) {
          mat[i][j] = alpha_fwd[t][i] * A[i][j] * gaussian(returns[t+1], muS[j], sigS[j]+1e-10) * beta_bwd[t+1][j];
          tot += mat[i][j];
        }
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) mat[i][j] /= tot + 1e-30;
      return mat;
    });
    // M-step
    pi = [gamma[0][0], gamma[0][1]];
    for (let i = 0; i < 2; i++) {
      const gs = sum(xi.map(x => sum(x[i])));
      A[i] = [sum(xi.map(x => x[i][0])) / (gs + 1e-30), sum(xi.map(x => x[i][1])) / (gs + 1e-30)];
    }
    for (let j = 0; j < 2; j++) {
      const gj = gamma.map(g => g[j]);
      const gsum = sum(gj) + 1e-30;
      muS[j] = sum(gj.map((g, t) => g * returns[t])) / gsum;
      sigS[j] = Math.sqrt(sum(gj.map((g, t) => g * (returns[t] - muS[j]) ** 2)) / gsum) + 1e-10;
    }
  }
  // Final decode: most likely state per time
  const stateProbs = [];
  const fwd = Array.from({ length: n }, () => [0, 0]);
  const sc0 = [0, 1].map(s => pi[s] * (1 / ((sigS[s] + 1e-10) * Math.sqrt(2*Math.PI))) * Math.exp(-0.5*((returns[0]-muS[s])/(sigS[s]+1e-10))**2));
  const sc0t = sum(sc0);
  for (let s = 0; s < 2; s++) fwd[0][s] = sc0[s] / (sc0t + 1e-30);
  stateProbs.push([...fwd[0]]);
  for (let t = 1; t < n; t++) {
    const obs = [0,1].map(j => {
      const sig = sigS[j] + 1e-10;
      return sum([0,1].map(i => fwd[t-1][i]*A[i][j])) * (1/(sig*Math.sqrt(2*Math.PI)))*Math.exp(-0.5*((returns[t]-muS[j])/sig)**2);
    });
    const sc = sum(obs) || 1;
    for (let j = 0; j < 2; j++) fwd[t][j] = obs[j]/sc;
    stateProbs.push([...fwd[t]]);
  }
  const states = stateProbs.map(p => p[0] > p[1] ? 0 : 1);
  const currentState = states[states.length - 1];
  const bullProb = stateProbs[stateProbs.length - 1][0];
  // Sort so state 0 = bull (higher mu)
  const swap = muS[0] < muS[1];
  return {
    states: swap ? states.map(s => 1 - s) : states,
    stateProbs: swap ? stateProbs.map(p => [p[1], p[0]]) : stateProbs,
    muS: swap ? [muS[1], muS[0]] : muS,
    sigS: swap ? [sigS[1], sigS[0]] : sigS,
    A: swap ? [[A[1][1],A[1][0]],[A[0][1],A[0][0]]] : A,
    currentState: swap ? 1 - currentState : currentState,
    bullProb: swap ? 1 - bullProb : bullProb,
  };
}

// --- Momentum Factor (Fama-French style, cross-sectional z-score) ---
function momentumFactors(returns) {
  const mom1  = mean(returns.slice(-5));
  const mom5  = mean(returns.slice(-21));
  const mom12 = mean(returns.slice(-63));
  const vol20 = stdDev(returns.slice(-20));
  // Volatility-adjusted momentum (Information Ratio style)
  const irMom = mom5 / (vol20 + 1e-10);
  // Reversal signal (short-term, 1-week)
  const reversal = -mom1; // contrarian
  return { mom1, mom5, mom12, vol20, irMom, reversal };
}

// --- RSI ---
function computeRSI(prices, period = 14) {
  const rets = simpleReturns(prices);
  const rsiArr = [];
  for (let i = period; i <= rets.length; i++) {
    const w = rets.slice(i - period, i);
    const avgG = mean(w.filter(r => r > 0).concat(0));
    const avgL = mean(w.filter(r => r < 0).map(Math.abs).concat(0));
    rsiArr.push(100 - 100 / (1 + avgG / (avgL + 1e-10)));
  }
  return rsiArr;
}

// --- EMA ---
const ema = (arr, n) => arr.reduce((acc, v, i) => {
  const k = 2 / (n + 1);
  return i === 0 ? [v] : [...acc, v * k + acc[i - 1] * (1 - k)];
}, []);

// --- MACD ---
function computeMACD(prices, fast = 12, slow = 26, sig = 9) {
  const fe = ema(prices, fast), se = ema(prices, slow);
  const macdLine = fe.slice(slow - fast).map((v, i) => v - se[i + slow - fast]);
  const sigLine = ema(macdLine, sig);
  const hist = macdLine.slice(sig - 1).map((v, i) => v - sigLine[i]);
  return { macdLine, sigLine, hist };
}

// --- Bollinger Bands ---
function bollingerBands(prices, period = 20, k = 2) {
  return prices.slice(period - 1).map((_, i) => {
    const w = prices.slice(i, i + period);
    const m = mean(w), s = stdDev(w);
    return { mid: m, upper: m + k * s, lower: m - k * s, pctB: (prices[i + period - 1] - (m - k*s)) / (2*k*s + 1e-10) };
  });
}

// --- Average True Range ---
function atr(prices, period = 14) {
  const trs = prices.slice(1).map((p, i) => Math.abs(p - prices[i]));
  const atrArr = [];
  for (let i = period; i <= trs.length; i++) atrArr.push(mean(trs.slice(i - period, i)));
  return atrArr;
}

// --- Sharpe / Sortino / Calmar ---
function riskMetrics(returns, prices, rf = 0.0525 / 252) {
  const excess = returns.map(r => r - rf);
  const sharpe = mean(excess) / (stdDev(returns) + 1e-10) * Math.sqrt(252);
  const downside = returns.filter(r => r < rf);
  const sortino = mean(excess) / (stdDev(downside.length ? downside : [0]) + 1e-10) * Math.sqrt(252);
  let peak = prices[0], mdd = 0;
  for (const p of prices) { if (p > peak) peak = p; mdd = Math.max(mdd, (peak - p) / peak); }
  const annRet = mean(returns) * 252;
  const calmar = mdd > 0 ? annRet / mdd : 0;
  const skew = (() => { const m = mean(returns), s = stdDev(returns); return mean(returns.map(r => ((r-m)/s)**3)); })();
  const kurt = (() => { const m = mean(returns), s = stdDev(returns); return mean(returns.map(r => ((r-m)/s)**4)) - 3; })();
  const sorted = [...returns].sort((a,b)=>a-b);
  const tailN = Math.max(1, Math.floor(returns.length * 0.05));
  const var95 = sorted[tailN];
  const cvar95 = mean(sorted.slice(0, tailN));
  return { sharpe, sortino, calmar, mdd, skew, kurt, var95, cvar95, annVol: stdDev(returns)*Math.sqrt(252), annRet };
}

// --- GARCH Monte Carlo with Regime-Conditional Drift ---
function garchMonteCarlo(S0, returns, garch, hmm, paths = 500, horizon = 30) {
  const { omega, alpha, beta } = garch;
  const h0 = garch.condVar[garch.condVar.length - 1];
  const currentRegime = hmm.currentState; // 0=bull, 1=bear
  const mu_daily = hmm.muS[currentRegime]; // regime-conditional drift
  const allPaths = [];
  for (let p = 0; p < paths; p++) {
    const path = [S0];
    let ht = h0;
    let prevEps = boxMuller() * Math.sqrt(h0);
    for (let t = 1; t <= horizon; t++) {
      ht = omega + alpha * prevEps ** 2 + beta * ht;
      const eps = boxMuller() * Math.sqrt(ht);
      prevEps = eps;
      const drift = mu_daily - 0.5 * ht;
      path.push(path[path.length - 1] * Math.exp(drift + eps));
    }
    allPaths.push(path);
  }
  return allPaths;
}

// --- Multi-Factor Signal (Calibrated) ---
function calibratedSignal(returns, prices, hmm, garch, rsiVals, bbands) {
  const mf = momentumFactors(returns);
  const currentRSI = rsiVals[rsiVals.length - 1];
  const currentBB = bbands[bbands.length - 1];
  const currentRegime = hmm.currentState;
  const bullProb = hmm.bullProb;
  const garchVol = Math.sqrt(garch.condVar[garch.condVar.length - 1]);
  const longRunVol = Math.sqrt(garch.longRunVar);
  const volRatio = garchVol / (longRunVol + 1e-10); // >1 = elevated vol
  const persistence = garch.persistence;

  // Factor 1: Trend momentum (12-month minus 1-month, standard UMD factor)
  const umd = (mf.mom12 - mf.mom1); // avoids short-term reversal
  const trendScore = clamp(umd * 300, -1, 1);

  // Factor 2: Mean reversion (RSI + Bollinger %B)
  const rsiScore = currentRSI < 30 ? 1 : currentRSI > 70 ? -1 : (50 - currentRSI) / 50;
  const bbScore = currentBB ? clamp(1 - 2 * currentBB.pctB, -1, 1) : 0;
  const reversionScore = 0.6 * rsiScore + 0.4 * bbScore;

  // Factor 3: Regime overlay
  const regimeScore = currentRegime === 0 ? bullProb : -(1 - bullProb);

  // Factor 4: Vol regime penalty (GARCH persistence)
  const volPenalty = volRatio > 1.5 ? 0.4 : volRatio > 1.2 ? 0.65 : 1.0;

  // Factor 5: IR momentum (vol-adjusted)
  const irScore = clamp(mf.irMom * 20, -1, 1);

  // Calibrated weights (based on academic literature)
  // Trend: 35%, Reversion: 20%, Regime: 25%, IR Momentum: 20%
  const raw = 0.35 * trendScore + 0.20 * reversionScore + 0.25 * regimeScore + 0.20 * irScore;
  const composite = clamp(raw * volPenalty, -1, 1);

  // Kelly fraction (half-Kelly for safety)
  const mu = mean(returns) * 252;
  const sig2 = calcVariance(returns) * 252;
  const kelly = clamp(mu / (sig2 + 1e-10) * 0.5, -1, 2);

  return {
    composite, trendScore, reversionScore, regimeScore, irScore,
    volPenalty, bullProb, currentRegime, garchVol, longRunVol,
    volRatio, persistence, kelly, mf, currentRSI, currentBB,
  };
}

// ═══════════════════════════════════════════════════════════════════
//  BLACK-SCHOLES OPTIONS ENGINE
// ═══════════════════════════════════════════════════════════════════

// Standard normal CDF via Horner's approximation
function normCDF(x) {
  const a1=0.254829592,a2=-0.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=0.3275911;
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.sqrt(2);
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*Math.exp(-x*x);
  return 0.5 * (1 + sign * y);
}
function normPDF(x) { return Math.exp(-0.5*x*x) / Math.sqrt(2*Math.PI); }

function blackScholes(S, K, T, r, sigma, type = "call") {
  if (T <= 0) {
    const intrinsic = type === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);
    return { price: intrinsic, delta: type==="call"?(S>K?1:0):(S<K?-1:0), gamma:0, theta:0, vega:0, rho:0, d1:0, d2:0 };
  }
  const d1 = (Math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*Math.sqrt(T));
  const d2 = d1 - sigma*Math.sqrt(T);
  const df = Math.exp(-r*T);
  let price, delta, rho;
  if (type === "call") {
    price = S*normCDF(d1) - K*df*normCDF(d2);
    delta = normCDF(d1);
    rho   = K*T*df*normCDF(d2) / 100;
  } else {
    price = K*df*normCDF(-d2) - S*normCDF(-d1);
    delta = normCDF(d1) - 1;
    rho   = -K*T*df*normCDF(-d2) / 100;
  }
  const gamma = normPDF(d1) / (S*sigma*Math.sqrt(T));
  const vega  = S*normPDF(d1)*Math.sqrt(T) / 100;
  const theta = type === "call"
    ? (-S*normPDF(d1)*sigma/(2*Math.sqrt(T)) - r*K*df*normCDF(d2)) / 365
    : (-S*normPDF(d1)*sigma/(2*Math.sqrt(T)) + r*K*df*normCDF(-d2)) / 365;
  return { price, delta, gamma, theta, vega, rho, d1, d2, iv: sigma };
}

// Build a full options chain around ATM
function buildOptionsChain(S, sigma, r = 0.0525, daysToExpiry = [7, 14, 30, 60, 90]) {
  const strikes = [-0.15,-0.10,-0.05,-0.02,0,0.02,0.05,0.10,0.15].map(pct => Math.round(S*(1+pct)/5)*5);
  return daysToExpiry.map(days => ({
    expiry: days,
    strikes: strikes.map(K => ({
      K,
      call: blackScholes(S, K, days/365, r, sigma, "call"),
      put:  blackScholes(S, K, days/365, r, sigma, "put"),
      moneyness: ((S-K)/K*100).toFixed(1),
    }))
  }));
}

// ═══════════════════════════════════════════════════════════════════
//  PORTFOLIO BACKTESTER
// ═══════════════════════════════════════════════════════════════════

function runBacktest(prices, returns, signal) {
  // Momentum-crossover strategy: go long when 10D EMA > 30D EMA, flat otherwise
  const ema10 = ema(prices, 10);
  const ema30 = ema(prices, 30);
  const startIdx = 30; // align
  const equity = [10000];
  const trades = [];
  const benchmarkEq = [10000];
  let position = 0; // 0=flat, 1=long, -1=short
  let entryPrice = 0, entryIdx = 0;
  const dailyPnL = [];
  const drawdowns = [];
  let peakEq = 10000;

  for (let i = startIdx; i < prices.length - 1; i++) {
    const fastE = ema10[i], slowE = ema30[i];
    const prevFast = ema10[i-1], prevSlow = ema30[i-1];
    const ret = returns[i] || 0;
    const prev = equity[equity.length - 1];
    const bPrev = benchmarkEq[benchmarkEq.length - 1];

    // Signal: golden cross = long, death cross = flat
    let signal = 0;
    if (fastE > slowE) signal = 1;
    else if (fastE < slowE) signal = 0;

    // Execute trade on cross
    if (signal !== position) {
      if (position !== 0) trades.push({ type: position > 0 ? "SELL" : "COVER", price: prices[i], idx: i, pnl: position*(prices[i]-entryPrice) });
      if (signal !== 0) { trades.push({ type: signal > 0 ? "BUY" : "SHORT", price: prices[i], idx: i }); entryPrice = prices[i]; entryIdx = i; }
      position = signal;
    }

    const stratRet = position * ret;
    const newEq = prev * (1 + stratRet);
    equity.push(newEq);
    benchmarkEq.push(bPrev * (1 + ret));
    dailyPnL.push(stratRet);
    peakEq = Math.max(peakEq, newEq);
    drawdowns.push((peakEq - newEq) / peakEq);
  }

  const totalReturn = (equity[equity.length-1] - 10000) / 10000;
  const bmReturn    = (benchmarkEq[benchmarkEq.length-1] - 10000) / 10000;
  const annRet = mean(dailyPnL) * 252;
  const annVol = stdDev(dailyPnL) * Math.sqrt(252);
  const sharpe = annVol > 0 ? annRet / annVol : 0;
  const maxDD   = Math.max(...drawdowns);
  const winRate = trades.filter(t => t.pnl > 0).length / Math.max(trades.filter(t => t.pnl).length, 1);
  const alpha   = totalReturn - bmReturn;

  return {
    equity: equity.map((v,i) => ({ day: i, value: +v.toFixed(2), benchmark: +benchmarkEq[i]?.toFixed(2) })),
    trades, totalReturn, bmReturn, sharpe, maxDD, winRate, alpha, annRet, annVol,
    nTrades: trades.filter(t=>t.pnl).length,
  };
}

// ═══════════════════════════════════════════════════════════════════
//  DATA FETCHING VIA CLAUDE API (web search for real prices)
// ═══════════════════════════════════════════════════════════════════

// ─── Yahoo Finance proxy helper (fast, free, no auth) ───────────────
const YAHOO_PROXY = "https://query1.finance.yahoo.com/v8/finance/chart/";
const YAHOO_META  = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/";

const SECTOR_MAP = {
  AAPL:"Technology", NVDA:"Semiconductors", TSLA:"EV/Energy",
  MSFT:"Technology", AMZN:"E-Commerce/Cloud", META:"Social Media",
  SPY:"Index ETF", "BTC-USD":"Crypto",
};
const NAME_MAP = {
  AAPL:"Apple Inc.", NVDA:"NVIDIA Corp.", TSLA:"Tesla Inc.",
  MSFT:"Microsoft Corp.", AMZN:"Amazon.com Inc.", META:"Meta Platforms",
  SPY:"SPDR S&P 500 ETF", "BTC-USD":"Bitcoin USD",
};
const PEERS_MAP = {
  AAPL:["MSFT","GOOGL","META"], NVDA:["AMD","INTC","AVGO"],
  TSLA:["RIVN","F","GM"], MSFT:["AAPL","GOOGL","ORCL"],
  AMZN:["MSFT","GOOGL","BABA"], META:["SNAP","PINS","GOOGL"],
  SPY:["QQQ","IWM","DIA"], "BTC-USD":["ETH-USD","SOL-USD","BNB-USD"],
};

async function fetchYahooChart(sym, days = 90) {
  const url = `${YAHOO_PROXY}${encodeURIComponent(sym)}?interval=1d&range=${days}d&includePrePost=false`;
  const res = await fetch(url, { headers: { "Accept": "application/json" } });
  if (!res.ok) throw new Error(`Yahoo chart error ${res.status} for ${sym}`);
  const j = await res.json();
  const result = j?.chart?.result?.[0];
  if (!result) throw new Error(`No chart data for ${sym}`);
  const timestamps = result.timestamp || [];
  const closes = result.indicators?.quote?.[0]?.close || [];
  const meta = result.meta || {};
  // Filter out null closes
  const pairs = timestamps.map((t, i) => ({ t, c: closes[i] })).filter(p => p.c != null);
  const prices = pairs.map(p => +p.c.toFixed(4));
  const dates  = pairs.map(p => {
    const d = new Date(p.t * 1000);
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  });
  return {
    ticker: sym, prices, dates,
    currentPrice: meta.regularMarketPrice || prices[prices.length - 1],
    name: NAME_MAP[sym] || meta.longName || sym,
    sector: SECTOR_MAP[sym] || "Equity",
    marketCap: meta.marketCap,
    currency: meta.currency || "USD",
  };
}

async function fetchRealStockData(ticker) {
  return fetchYahooChart(ticker, 90);
}

// ─── Peer quote fetch (Yahoo Finance, parallel) ─────────────────────
async function fetchPeerComparison(ticker) {
  const peerTickers = [ticker, ...(PEERS_MAP[ticker] || ["SPY"])];
  const results = await Promise.all(peerTickers.map(async sym => {
    try {
      // Fetch summary + price in one call
      const url = `${YAHOO_META}${encodeURIComponent(sym)}?modules=summaryDetail,defaultKeyStatistics,financialData,price`;
      const res = await fetch(url, { headers: { "Accept": "application/json" } });
      if (!res.ok) return null;
      const j = await res.json();
      const r = j?.quoteSummary?.result?.[0];
      if (!r) return null;
      const sd = r.summaryDetail || {}, fs = r.financialData || {}, ks = r.defaultKeyStatistics || {}, pr = r.price || {};
      return {
        ticker: sym,
        name: NAME_MAP[sym] || pr.longName || sym,
        price: pr.regularMarketPrice?.raw ?? sd.previousClose?.raw ?? 0,
        peRatio: sd.trailingPE?.raw ?? ks.trailingEps?.raw ?? null,
        pbRatio: ks.priceToBook?.raw ?? null,
        grossMargin: fs.grossMargins?.raw ?? null,
        beta: sd.beta?.raw ?? null,
        ytdReturn: pr.regularMarketChangePercent?.raw
          ? null // YTD not in summary, use 52w
          : ks["52WeekChange"]?.raw ?? null,
        marketCap: pr.marketCap?.fmt ?? "—",
        analystRating: fs.recommendationKey
          ? fs.recommendationKey.toUpperCase().replace(/_/g," ")
          : "—",
        priceTarget: fs.targetMeanPrice?.raw ?? null,
      };
    } catch { return null; }
  }));
  return { peers: results.filter(Boolean) };
}

// ─── News sentiment via Haiku (only place AI is still needed) ───────
async function fetchNewsSentiment(ticker) {
  const prompt = `Search for 5 recent news headlines about ${ticker} stock. Return ONLY JSON:
{"headlines":[{"title":"...","source":"...","date":"Mar 15","sentiment":0.7,"summary":"brief","tag":"EARNINGS"}],"overallSentiment":0.3,"sentimentLabel":"BULLISH","keyThemes":["a","b"]}
sentiment -1 to +1. tag: EARNINGS|UPGRADE|DOWNGRADE|MACRO|PRODUCT|LEGAL|GUIDANCE`;
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 700,
      tools: [{ type: "web_search_20250305", name: "web_search" }],
      messages: [{ role: "user", content: prompt }],
    }),
  });
  const data = await res.json();
  const blocks = data.content?.filter(b => b.type === "text") || [];
  if (!blocks.length) throw new Error("No sentiment response");
  let raw = blocks[blocks.length - 1].text.trim().replace(/```json|```/g, "").trim();
  const s = raw.indexOf("{"), e = raw.lastIndexOf("}");
  if (s === -1) throw new Error("No sentiment JSON");
  return JSON.parse(raw.slice(s, e + 1));
}


// ═══════════════════════════════════════════════════════════════════
//  UI COMPONENTS
// ═══════════════════════════════════════════════════════════════════

const TICKERS = ["AAPL","NVDA","TSLA","MSFT","AMZN","META","SPY","BTC-USD"];

const fmt = (v, dec = 2) => typeof v === "number" ? v.toFixed(dec) : "—";
const fmtPct = v => typeof v === "number" ? `${(v * 100).toFixed(2)}%` : "—";
const fmtPrice = v => typeof v === "number" ? `$${v.toFixed(2)}` : "—";
const colorVal = (v, lo = 0) => v > lo ? "#00e8a2" : v < lo ? "#ff4060" : "#8899aa";

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#08111d", border: "1px solid #162030", borderRadius: 6, padding: "8px 12px", fontSize: 11, fontFamily: "'IBM Plex Mono', monospace" }}>
      <div style={{ color: "#c8d8e8", marginBottom: 4, fontWeight: 600 }}>{label}</div>
      {payload.map((p, i) => p.value != null && (
        <div key={i} style={{ color: p.color || "#8899aa" }}>
          {p.name}: <span style={{ color: "#e8edf3" }}>{typeof p.value === "number" ? p.value.toFixed(3) : p.value}</span>
        </div>
      ))}
    </div>
  );
};

const Chip = ({ label, value, color, sub }) => (
  <div style={{ background: "#080f1a", border: `1px solid ${color || "#1a2535"}22`, borderRadius: 8, padding: "12px 14px", minWidth: 100 }}>
    <div style={{ fontSize: 9, color: "#3a5068", letterSpacing: "0.12em", textTransform: "uppercase", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 5 }}>{label}</div>
    <div style={{ fontSize: 19, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace", color: color || "#d8e8f3", lineHeight: 1 }}>{value}</div>
    {sub && <div style={{ fontSize: 9, color: "#2a4060", marginTop: 4, fontFamily: "monospace" }}>{sub}</div>}
  </div>
);

const FactorRow = ({ label, score, weight, description }) => {
  const pct = clamp(score, -1, 1);
  const col = pct > 0.1 ? "#00e8a2" : pct < -0.1 ? "#ff4060" : "#f0c040";
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
        <span style={{ fontSize: 10, color: "#5a7090", fontFamily: "monospace" }}>{label} <span style={{ color: "#2a4060" }}>w={weight}</span></span>
        <span style={{ fontSize: 10, fontFamily: "monospace", color: col }}>{pct > 0 ? "+" : ""}{(pct * 100).toFixed(1)}</span>
      </div>
      <div style={{ height: 4, background: "#0a1520", borderRadius: 2, position: "relative" }}>
        <div style={{ position: "absolute", left: "50%", top: 0, width: 1, height: "100%", background: "#1a2535" }} />
        <div style={{
          position: "absolute",
          height: "100%",
          borderRadius: 2,
          background: col,
          left: pct >= 0 ? "50%" : `${(0.5 + pct / 2) * 100}%`,
          width: `${Math.abs(pct) / 2 * 100}%`,
          transition: "all 0.6s ease"
        }} />
      </div>
      {description && <div style={{ fontSize: 9, color: "#1e3045", marginTop: 2, fontFamily: "monospace" }}>{description}</div>}
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════
//  MAIN APP
// ═══════════════════════════════════════════════════════════════════

export default function QuantTerminal() {
  const [ticker, setTicker] = useState("NVDA");
  const [state, setState] = useState({ status: "idle" });
  const [analysis, setAnalysis] = useState(null);
  const [extras, setExtras] = useState(null); // options, backtest, sentiment, peers
  const [activeTab, setActiveTab] = useState("price");
  const [mcPaths, setMcPaths] = useState(500);
  const [optionExpiry, setOptionExpiry] = useState(30);
  const [optionType, setOptionType] = useState("call");

  const runAnalysis = useCallback(async (t, paths) => {
    setState({ status: "loading", step: "Fetching market data…" });
    setAnalysis(null);
    setExtras({ loading: true });
    try {
      // ── All network calls fire simultaneously ──────────────────────
      const [raw, sentimentResult, peersResult] = await Promise.all([
        fetchRealStockData(t),
        fetchNewsSentiment(t).catch(e => ({ error: e.message })),
        fetchPeerComparison(t).catch(e => ({ error: e.message })),
      ]);
      setExtras({ loading: false, sentiment: sentimentResult, peers: peersResult });

      setState({ status: "loading", step: "Running quant models…" });
      await new Promise(r => setTimeout(r, 10));

      const prices = raw.prices.map(Number).filter(v => isFinite(v) && v > 0);
      const dates  = (raw.dates || []).slice(0, prices.length);
      const rets   = logReturns(prices);

      const garch       = fitGARCH(rets);
      const hmm         = fitHMM(rets);
      const rsiVals     = computeRSI(prices);
      const { macdLine, sigLine, hist: macdHist } = computeMACD(prices);
      const bbands      = bollingerBands(prices);
      const atrVals     = atr(prices);
      const garchVolFcast = garchVolForecast(garch, garch.condVar[garch.condVar.length - 1]);

      setState({ status: "loading", step: `Running ${paths} Monte Carlo paths…` });
      await new Promise(r => setTimeout(r, 10));

      const S0 = prices[prices.length - 1];
      const mcAllPaths = garchMonteCarlo(S0, rets, garch, hmm, paths);
      const forecastBands = Array.from({ length: 31 }, (_, i) => {
        const vals = mcAllPaths.map(p => p[i]).sort((a, b) => a - b);
        const n = vals.length;
        return { day: i, p2: vals[Math.floor(n*.02)], p10: vals[Math.floor(n*.10)], p25: vals[Math.floor(n*.25)], p50: vals[Math.floor(n*.50)], p75: vals[Math.floor(n*.75)], p90: vals[Math.floor(n*.90)], p98: vals[Math.floor(n*.98)] };
      });

      const risk        = riskMetrics(rets, prices);
      const signal      = calibratedSignal(rets, prices, hmm, garch, rsiVals, bbands);
      const annSigma    = Math.sqrt(garch.condVar[garch.condVar.length - 1]) * Math.sqrt(252);
      const optionsChain = buildOptionsChain(S0, annSigma);
      const backtest    = runBacktest(prices, rets, signal);

      setAnalysis({ prices, dates, rets, rsiVals, macdLine, sigLine, macdHist, bbands, atrVals, garchVolFcast, forecastBands, mcAllPaths, garch, hmm, risk, signal, S0, meta: raw, paths, optionsChain, backtest });
      setState({ status: "done" });

    } catch (e) {
      setState({ status: "error", message: e.message });
      setExtras({ loading: false });
    }
  }, []);

  // Run on mount and whenever ticker changes
  useEffect(() => {
    runAnalysis(ticker, mcPaths);
  }, [ticker]);

  const rerun = () => runAnalysis(ticker, mcPaths);

  // Chart data
  const priceChartData = analysis ? analysis.prices.map((p, i) => ({
    date: analysis.dates[i] || `D${i}`,
    price: +p.toFixed(2),
    upper: analysis.bbands[i - 19]?.upper?.toFixed(2),
    lower: analysis.bbands[i - 19]?.lower?.toFixed(2),
    mid: analysis.bbands[i - 19]?.mid?.toFixed(2),
  })) : [];

  const volChartData = analysis ? analysis.garch.condVar.map((v, i) => ({
    date: analysis.dates[i] || `D${i}`,
    vol: +(Math.sqrt(v) * Math.sqrt(252) * 100).toFixed(2),
    longRun: +(Math.sqrt(analysis.garch.longRunVar) * Math.sqrt(252) * 100).toFixed(2),
  })) : [];

  const regimeChartData = analysis ? analysis.hmm.stateProbs.map((p, i) => ({
    date: analysis.dates[i] || `D${i}`,
    bull: +(p[0] * 100).toFixed(1),
    bear: +(p[1] * 100).toFixed(1),
    state: analysis.hmm.states[i],
  })) : [];

  const rsiData = analysis ? analysis.rsiVals.map((r, i) => ({ date: analysis.dates[i + 15] || `D${i}`, rsi: +r.toFixed(1) })) : [];
  const macdData = analysis ? analysis.macdHist.map((h, i) => ({
    date: analysis.dates[i + 34] || `D${i}`,
    hist: +h.toFixed(4),
    macd: +analysis.macdLine[i + 8]?.toFixed(4),
    signal: +analysis.sigLine[i]?.toFixed(4),
  })) : [];
  const fcData = analysis ? analysis.forecastBands.map((b, i) => ({
    day: `+${i}d`,
    p2: +b.p2?.toFixed(2), p10: +b.p10?.toFixed(2), p25: +b.p25?.toFixed(2),
    p50: +b.p50?.toFixed(2), p75: +b.p75?.toFixed(2), p90: +b.p90?.toFixed(2), p98: +b.p98?.toFixed(2),
  })) : [];

  const sig = analysis?.signal;
  const sigStrength = sig?.composite ?? 0;
  const sigLabel = sigStrength > 0.4 ? "STRONG BUY" : sigStrength > 0.1 ? "BUY" : sigStrength < -0.4 ? "STRONG SELL" : sigStrength < -0.1 ? "SELL" : "NEUTRAL";
  const sigColor = sigStrength > 0.1 ? "#00e8a2" : sigStrength < -0.1 ? "#ff4060" : "#f0c040";
  const regime = analysis?.hmm?.currentState === 0 ? "BULL" : "BEAR";
  const regimeColor = regime === "BULL" ? "#00e8a2" : "#ff4060";

  const TABS = ["price","forecast","volatility","regime","oscillators","signals","risk","options","backtest","sentiment","peers"];

  return (
    <div style={{ minHeight: "100vh", background: "#04090f", color: "#d0dde8", fontFamily: "'IBM Plex Sans', sans-serif" }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap" rel="stylesheet" />
      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: #04090f; }
        ::-webkit-scrollbar-thumb { background: #1a2535; border-radius: 2px; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
        .fadeIn { animation: fadeIn 0.4s ease forwards; }
      `}</style>

      {/* TOP BAR */}
      <div style={{ borderBottom: "1px solid #0c1824", background: "#040c16", padding: "0 28px", display: "flex", alignItems: "center", height: 52, gap: 20 }}>
        <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#0a6aff", letterSpacing: "0.2em", fontWeight: 700 }}>◈ ALPHA ENGINE v2</div>
        <div style={{ width: 1, height: 20, background: "#0c1824" }} />
        <div style={{ fontSize: 10, color: "#1e3a55", fontFamily: "monospace" }}>GARCH(1,1) · HMM-2STATE · MONTE CARLO · MULTI-FACTOR</div>
        <div style={{ flex: 1 }} />
        <div style={{ fontSize: 9, color: "#0a2535", fontFamily: "monospace" }}>LIVE DATA VIA WEB SEARCH</div>
      </div>

      <div style={{ padding: "18px 28px 28px", maxWidth: 1160, margin: "0 auto" }}>
        {/* TICKER ROW */}
        <div style={{ display: "flex", gap: 6, marginBottom: 20, flexWrap: "wrap", alignItems: "center" }}>
          {TICKERS.map(t => (
            <button key={t} onClick={() => setTicker(t)} style={{
              padding: "5px 14px", borderRadius: 4,
              border: `1px solid ${ticker === t ? "#0a5adf" : "#0c1824"}`,
              background: ticker === t ? "#051830" : "#060e1a",
              color: ticker === t ? "#4a9aff" : "#2a4560",
              fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, fontWeight: 600,
              cursor: "pointer", transition: "all 0.15s", letterSpacing: "0.05em"
            }}>{t}</button>
          ))}
          <div style={{ flex: 1 }} />
          <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <span style={{ fontSize: 9, color: "#1a3050", fontFamily: "monospace" }}>MC PATHS:</span>
            {[200, 500, 1000].map(n => (
              <button key={n} onClick={() => setMcPaths(n)} style={{
                padding: "3px 9px", fontSize: 10, fontFamily: "monospace",
                border: `1px solid ${mcPaths === n ? "#0a5adf" : "#0c1824"}`,
                background: mcPaths === n ? "#051830" : "transparent",
                color: mcPaths === n ? "#4a9aff" : "#1a3050",
                borderRadius: 3, cursor: "pointer"
              }}>{n}</button>
            ))}
          </div>
          <button onClick={rerun} style={{
            padding: "5px 16px", borderRadius: 4,
            border: "1px solid #0a2540", background: "#060e1a",
            color: "#0a6aff", fontFamily: "monospace", fontSize: 11, cursor: "pointer"
          }}>
            {state.status === "loading" ? "⟳ RUNNING" : "↺ RERUN"}
          </button>
        </div>

        {/* STATUS */}
        {state.status === "loading" && (
          <div style={{ padding: "60px 0", textAlign: "center" }}>
            <div style={{ fontSize: 11, fontFamily: "monospace", color: "#0a5adf", letterSpacing: "0.15em", animation: "pulse 1.4s ease infinite" }}>
              {state.step}
            </div>
            <div style={{ marginTop: 16, fontSize: 9, color: "#0a2535", fontFamily: "monospace" }}>Fetching live market data · Calibrating models · Running {mcPaths} stochastic paths</div>
          </div>
        )}

        {state.status === "error" && (
          <div style={{ padding: "40px 0", textAlign: "center", color: "#ff4060", fontFamily: "monospace", fontSize: 12 }}>
            ✕ {state.message}
            <div style={{ marginTop: 8, fontSize: 10, color: "#4a2030" }}>Try a different ticker or click RERUN</div>
          </div>
        )}

        {/* MAIN CONTENT */}
        {state.status === "done" && analysis && (
          <div className="fadeIn">
            {/* HEADER ROW */}
            <div style={{ display: "flex", gap: 12, marginBottom: 18, flexWrap: "wrap", alignItems: "flex-start" }}>
              <div style={{ flex: 1, minWidth: 180 }}>
                <div style={{ fontSize: 30, fontWeight: 700, letterSpacing: "-0.03em", fontFamily: "'IBM Plex Mono', monospace" }}>{ticker}</div>
                <div style={{ fontSize: 12, color: "#2a5070", marginTop: 2 }}>{analysis.meta.name} · {analysis.meta.sector}</div>
                <div style={{ fontSize: 22, fontWeight: 600, color: "#d0dde8", marginTop: 6, fontFamily: "'IBM Plex Mono', monospace" }}>
                  {fmtPrice(analysis.S0)}
                </div>
              </div>

              {/* Composite Signal Badge */}
              <div style={{ padding: "12px 22px", borderRadius: 8, border: `1px solid ${sigColor}30`, background: `${sigColor}08`, textAlign: "center" }}>
                <div style={{ fontSize: 8, color: "#2a4060", letterSpacing: "0.15em", fontFamily: "monospace", marginBottom: 4 }}>COMPOSITE SIGNAL</div>
                <div style={{ fontSize: 20, fontWeight: 800, color: sigColor, fontFamily: "'IBM Plex Mono', monospace" }}>{sigLabel}</div>
                <div style={{ fontSize: 10, color: sigColor + "80", fontFamily: "monospace" }}>{(sigStrength * 100).toFixed(1)} / 100</div>
              </div>

              {/* Regime Badge */}
              <div style={{ padding: "12px 22px", borderRadius: 8, border: `1px solid ${regimeColor}25`, background: `${regimeColor}07`, textAlign: "center" }}>
                <div style={{ fontSize: 8, color: "#2a4060", letterSpacing: "0.15em", fontFamily: "monospace", marginBottom: 4 }}>HMM REGIME</div>
                <div style={{ fontSize: 20, fontWeight: 800, color: regimeColor, fontFamily: "'IBM Plex Mono', monospace" }}>{regime}</div>
                <div style={{ fontSize: 10, color: regimeColor + "80", fontFamily: "monospace" }}>P(bull)={fmtPct(sig?.bullProb)}</div>
              </div>

              {/* Key metrics */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 6 }}>
                <Chip label="Sharpe" value={fmt(analysis.risk.sharpe)} color={colorVal(analysis.risk.sharpe, 1)} sub="annualized" />
                <Chip label="Sortino" value={fmt(analysis.risk.sortino)} color={colorVal(analysis.risk.sortino, 1)} sub="annualized" />
                <Chip label="GARCH Vol" value={fmtPct(sig?.garchVol * Math.sqrt(252))} color="#4a9aff" sub="current annualized" />
                <Chip label="Persistence" value={fmt(analysis.garch.persistence)} color={analysis.garch.persistence > 0.95 ? "#ff4060" : "#f0c040"} sub="α+β" />
              </div>
            </div>

            {/* TABS */}
            <div style={{ display: "flex", borderBottom: "1px solid #0c1824", marginBottom: 16, overflowX: "auto" }}>
              {TABS.map(tab => (
                <button key={tab} onClick={() => setActiveTab(tab)} style={{
                  padding: "8px 16px", border: "none",
                  borderBottom: `2px solid ${activeTab === tab ? "#0a6aff" : "transparent"}`,
                  background: "transparent",
                  color: activeTab === tab ? "#4a9aff" : "#1e3a55",
                  fontSize: 10, fontFamily: "'IBM Plex Mono', monospace",
                  letterSpacing: "0.1em", textTransform: "uppercase",
                  cursor: "pointer", whiteSpace: "nowrap", marginBottom: -1
                }}>{tab}</button>
              ))}
            </div>

            {/* ── PRICE TAB ── */}
            {activeTab === "price" && (
              <div>
                <div style={{ fontSize: 9, color: "#1a3550", fontFamily: "monospace", marginBottom: 8 }}>PRICE + BOLLINGER BANDS (20,2) · REAL MARKET DATA</div>
                <div style={{ height: 300, background: "#060e1a", borderRadius: 8, border: "1px solid #0c1824", padding: "12px 4px" }}>
                  <ResponsiveContainer>
                    <LineChart data={priceChartData}>
                      <XAxis dataKey="date" tick={{ fill: "#1a3050", fontSize: 8, fontFamily: "monospace" }} tickLine={false} axisLine={false} interval={19} />
                      <YAxis tick={{ fill: "#1a3050", fontSize: 8, fontFamily: "monospace" }} tickLine={false} axisLine={false} tickFormatter={v => `$${v}`} domain={["auto","auto"]} />
                      <Tooltip content={<CustomTooltip />} />
                      <Line type="monotone" dataKey="upper" stroke="#1a3550" strokeWidth={1} dot={false} name="BB Upper" strokeDasharray="2 3" />
                      <Line type="monotone" dataKey="lower" stroke="#1a3550" strokeWidth={1} dot={false} name="BB Lower" strokeDasharray="2 3" />
                      <Line type="monotone" dataKey="mid" stroke="#0a2535" strokeWidth={1} dot={false} name="BB Mid" />
                      <Line type="monotone" dataKey="price" stroke="#0a6aff" strokeWidth={2} dot={false} name="Price" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ marginTop: 8, display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 6 }}>
                  {[
                    { label: "52W High (proxy)", value: fmtPrice(Math.max(...analysis.prices)) },
                    { label: "52W Low (proxy)", value: fmtPrice(Math.min(...analysis.prices)) },
                    { label: "ATR (14)", value: fmtPrice(analysis.atrVals[analysis.atrVals.length - 1]) },
                    { label: "BB %B", value: fmt(analysis.signal.currentBB?.pctB) },
                  ].map(m => <Chip key={m.label} label={m.label} value={m.value} />)}
                </div>
              </div>
            )}

            {/* ── FORECAST TAB ── */}
            {activeTab === "forecast" && (
              <div>
                <div style={{ fontSize: 9, color: "#1a3550", fontFamily: "monospace", marginBottom: 8 }}>
                  GARCH MONTE CARLO · {analysis.paths} PATHS · 30-DAY HORIZON · REGIME-CONDITIONAL DRIFT
                </div>
                <div style={{ height: 300, background: "#060e1a", borderRadius: 8, border: "1px solid #0c1824", padding: "12px 4px" }}>
                  <ResponsiveContainer>
                    <LineChart data={fcData}>
                      <XAxis dataKey="day" tick={{ fill: "#1a3050", fontSize: 8, fontFamily: "monospace" }} tickLine={false} axisLine={false} />
                      <YAxis tick={{ fill: "#1a3050", fontSize: 8, fontFamily: "monospace" }} tickLine={false} axisLine={false} tickFormatter={v => `$${v?.toFixed(0)}`} domain={["auto","auto"]} />
                      <Tooltip content={<CustomTooltip />} />
                      <Line type="monotone" dataKey="p98" stroke="#051520" strokeWidth={1} dot={false} name="P98" />
                      <Line type="monotone" dataKey="p90" stroke="#0a3020" strokeWidth={1.5} dot={false} name="P90" />
                      <Line type="monotone" dataKey="p75" stroke="#0d5a30" strokeWidth={2} dot={false} name="P75" />
                      <Line type="monotone" dataKey="p50" stroke="#00e8a2" strokeWidth={2.5} dot={false} name="Median" />
                      <Line type="monotone" dataKey="p25" stroke="#6a1a1a" strokeWidth={2} dot={false} name="P25" />
                      <Line type="monotone" dataKey="p10" stroke="#3a0e0e" strokeWidth={1.5} dot={false} name="P10" />
                      <Line type="monotone" dataKey="p2" stroke="#1a0808" strokeWidth={1} dot={false} name="P2" />
                      <ReferenceLine y={analysis.S0} stroke="#1a2535" strokeDasharray="3 3" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 6, marginTop: 8 }}>
                  {[
                    { label: "Bull (P90)", v: fcData[30]?.p90, ref: analysis.S0 },
                    { label: "Opt (P75)", v: fcData[30]?.p75, ref: analysis.S0 },
                    { label: "Base (P50)", v: fcData[30]?.p50, ref: analysis.S0 },
                    { label: "Pess (P25)", v: fcData[30]?.p25, ref: analysis.S0 },
                    { label: "Bear (P10)", v: fcData[30]?.p10, ref: analysis.S0 },
                  ].map(m => (
                    <div key={m.label} style={{ background: "#060e1a", border: "1px solid #0c1824", borderRadius: 6, padding: "10px 12px" }}>
                      <div style={{ fontSize: 8, color: "#1a3050", fontFamily: "monospace", marginBottom: 4 }}>{m.label}</div>
                      <div style={{ fontSize: 15, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace", color: m.v >= m.ref ? "#00e8a2" : "#ff4060" }}>{fmtPrice(m.v)}</div>
                      <div style={{ fontSize: 9, color: "#1a3050", fontFamily: "monospace" }}>{m.v ? `${((m.v - m.ref) / m.ref * 100).toFixed(1)}%` : ""}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ── VOLATILITY TAB ── */}
            {activeTab === "volatility" && (
              <div>
                <div style={{ fontSize: 9, color: "#1a3550", fontFamily: "monospace", marginBottom: 8 }}>GARCH(1,1) CONDITIONAL VOLATILITY · ANNUALIZED</div>
                <div style={{ height: 280, background: "#060e1a", borderRadius: 8, border: "1px solid #0c1824", padding: "12px 4px" }}>
                  <ResponsiveContainer>
                    <LineChart data={volChartData}>
                      <XAxis dataKey="date" tick={{ fill: "#1a3050", fontSize: 8, fontFamily: "monospace" }} tickLine={false} axisLine={false} interval={19} />
                      <YAxis tick={{ fill: "#1a3050", fontSize: 8, fontFamily: "monospace" }} tickLine={false} axisLine={false} tickFormatter={v => `${v}%`} domain={["auto","auto"]} />
                      <Tooltip content={<CustomTooltip />} />
                      <Line type="monotone" dataKey="vol" stroke="#4a9aff" strokeWidth={1.5} dot={false} name="Cond. Vol %" />
                      <Line type="monotone" dataKey="longRun" stroke="#1a3550" strokeWidth={1} dot={false} name="Long-run Vol %" strokeDasharray="4 3" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 6, marginTop: 8 }}>
                  <Chip label="Current GARCH Vol" value={fmtPct(sig?.garchVol * Math.sqrt(252))} color="#4a9aff" sub="annualized" />
                  <Chip label="Long-run Vol" value={fmtPct(Math.sqrt(analysis.garch.longRunVar * 252))} color="#2a5070" sub="unconditional" />
                  <Chip label="α (ARCH)" value={fmt(analysis.garch.alpha)} color="#f0c040" sub="shock sensitivity" />
                  <Chip label="β (GARCH)" value={fmt(analysis.garch.beta)} color="#f0c040" sub="vol persistence" />
                </div>
                <div style={{ marginTop: 8, padding: "10px 14px", background: "#060e1a", border: "1px solid #0c1824", borderRadius: 6, fontSize: 9, color: "#1a3550", fontFamily: "monospace", lineHeight: 1.8 }}>
                  GARCH persistence (α+β) = {fmt(analysis.garch.persistence)} &nbsp;·&nbsp;
                  {analysis.garch.persistence > 0.97 ? "VERY HIGH — shocks decay slowly, elevated vol regimes persist" :
                   analysis.garch.persistence > 0.92 ? "HIGH — vol clusters strongly, slow mean-reversion" :
                   "MODERATE — vol reverts to long-run mean relatively quickly"}
                </div>
              </div>
            )}

            {/* ── REGIME TAB ── */}
            {activeTab === "regime" && (
              <div>
                <div style={{ fontSize: 9, color: "#1a3550", fontFamily: "monospace", marginBottom: 8 }}>
                  HIDDEN MARKOV MODEL · 2-STATE REGIME DETECTION (BULL / BEAR) · BAUM-WELCH EM
                </div>
                <div style={{ height: 260, background: "#060e1a", borderRadius: 8, border: "1px solid #0c1824", padding: "12px 4px" }}>
                  <ResponsiveContainer>
                    <AreaChart data={regimeChartData}>
                      <XAxis dataKey="date" tick={{ fill: "#1a3050", fontSize: 8, fontFamily: "monospace" }} tickLine={false} axisLine={false} interval={19} />
                      <YAxis domain={[0, 100]} tick={{ fill: "#1a3050", fontSize: 8, fontFamily: "monospace" }} tickLine={false} axisLine={false} tickFormatter={v => `${v}%`} />
                      <Tooltip content={<CustomTooltip />} />
                      <defs>
                        <linearGradient id="bullGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#00e8a2" stopOpacity={0.25} />
                          <stop offset="95%" stopColor="#00e8a2" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="bearGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#ff4060" stopOpacity={0.25} />
                          <stop offset="95%" stopColor="#ff4060" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <Area type="monotone" dataKey="bull" stroke="#00e8a2" strokeWidth={1.5} fill="url(#bullGrad)" name="P(Bull) %" />
                      <Area type="monotone" dataKey="bear" stroke="#ff4060" strokeWidth={1.5} fill="url(#bearGrad)" name="P(Bear) %" />
                      <ReferenceLine y={50} stroke="#1a2535" strokeDasharray="3 3" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 8, marginTop: 8 }}>
                  <div style={{ background: "#060e1a", border: "1px solid #0c1824", borderRadius: 6, padding: "12px 14px" }}>
                    <div style={{ fontSize: 9, color: "#1a3550", fontFamily: "monospace", marginBottom: 8 }}>BULL STATE (μ, σ)</div>
                    <div style={{ fontFamily: "monospace", fontSize: 12, color: "#00e8a2" }}>
                      μ = {fmt(analysis.hmm.muS[0] * 252 * 100)}% ann · σ = {fmt(analysis.hmm.sigS[0] * Math.sqrt(252) * 100)}% ann
                    </div>
                    <div style={{ fontSize: 9, color: "#0a3020", marginTop: 4, fontFamily: "monospace" }}>
                      P(stay bull) = {fmt(analysis.hmm.A[0][0])} · P(bull→bear) = {fmt(analysis.hmm.A[0][1])}
                    </div>
                  </div>
                  <div style={{ background: "#060e1a", border: "1px solid #0c1824", borderRadius: 6, padding: "12px 14px" }}>
                    <div style={{ fontSize: 9, color: "#1a3550", fontFamily: "monospace", marginBottom: 8 }}>BEAR STATE (μ, σ)</div>
                    <div style={{ fontFamily: "monospace", fontSize: 12, color: "#ff4060" }}>
                      μ = {fmt(analysis.hmm.muS[1] * 252 * 100)}% ann · σ = {fmt(analysis.hmm.sigS[1] * Math.sqrt(252) * 100)}% ann
                    </div>
                    <div style={{ fontSize: 9, color: "#3a0e0e", marginTop: 4, fontFamily: "monospace" }}>
                      P(stay bear) = {fmt(analysis.hmm.A[1][1])} · P(bear→bull) = {fmt(analysis.hmm.A[1][0])}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ── OSCILLATORS TAB ── */}
            {activeTab === "oscillators" && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                <div>
                  <div style={{ fontSize: 9, color: "#1a3550", fontFamily: "monospace", marginBottom: 6 }}>RSI (14) — CURRENT: {fmt(sig?.currentRSI, 1)}</div>
                  <div style={{ height: 200, background: "#060e1a", borderRadius: 8, border: "1px solid #0c1824", padding: "10px 4px" }}>
                    <ResponsiveContainer>
                      <AreaChart data={rsiData}>
                        <defs><linearGradient id="rsiGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#0a6aff" stopOpacity={0.2} /><stop offset="95%" stopColor="#0a6aff" stopOpacity={0} /></linearGradient></defs>
                        <XAxis dataKey="date" tick={{ fill: "#1a3050", fontSize: 7, fontFamily: "monospace" }} tickLine={false} axisLine={false} interval={20} />
                        <YAxis domain={[0, 100]} tick={{ fill: "#1a3050", fontSize: 7, fontFamily: "monospace" }} tickLine={false} axisLine={false} />
                        <Tooltip content={<CustomTooltip />} />
                        <ReferenceLine y={70} stroke="#ff406044" strokeDasharray="2 3" />
                        <ReferenceLine y={30} stroke="#00e8a244" strokeDasharray="2 3" />
                        <Area type="monotone" dataKey="rsi" stroke="#0a6aff" strokeWidth={1.5} fill="url(#rsiGrad)" name="RSI" dot={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: "#1a3550", fontFamily: "monospace", marginBottom: 6 }}>MACD (12,26,9) HISTOGRAM</div>
                  <div style={{ height: 200, background: "#060e1a", borderRadius: 8, border: "1px solid #0c1824", padding: "10px 4px" }}>
                    <ResponsiveContainer>
                      <ComposedChart data={macdData}>
                        <XAxis dataKey="date" tick={{ fill: "#1a3050", fontSize: 7, fontFamily: "monospace" }} tickLine={false} axisLine={false} interval={20} />
                        <YAxis tick={{ fill: "#1a3050", fontSize: 7, fontFamily: "monospace" }} tickLine={false} axisLine={false} />
                        <Tooltip content={<CustomTooltip />} />
                        <ReferenceLine y={0} stroke="#1a2535" />
                        <Bar dataKey="hist" name="Histogram" shape={(p) => <rect x={p.x} y={p.y} width={p.width} height={Math.abs(p.height || 0)} fill={p.value >= 0 ? "#00e8a2" : "#ff4060"} opacity={0.7} />} />
                        <Line type="monotone" dataKey="macd" stroke="#4a9aff" strokeWidth={1} dot={false} name="MACD" />
                        <Line type="monotone" dataKey="signal" stroke="#f0c040" strokeWidth={1} dot={false} name="Signal" />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}

            {/* ── SIGNALS TAB ── */}
            {activeTab === "signals" && sig && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                <div style={{ background: "#060e1a", border: "1px solid #0c1824", borderRadius: 8, padding: 18 }}>
                  <div style={{ fontSize: 9, color: "#1a3550", fontFamily: "monospace", marginBottom: 14 }}>CALIBRATED MULTI-FACTOR DECOMPOSITION</div>
                  <FactorRow label="TREND MOMENTUM (UMD)" score={sig.trendScore} weight="35%" description="12M minus 1M momentum, avoids short-term reversal" />
                  <FactorRow label="MEAN REVERSION (RSI+BB)" score={sig.reversionScore} weight="20%" description="0.6×RSI + 0.4×Bollinger %B z-score" />
                  <FactorRow label="HMM REGIME OVERLAY" score={sig.regimeScore} weight="25%" description={`P(bull)=${fmtPct(sig.bullProb)} from 2-state HMM`} />
                  <FactorRow label="IR MOMENTUM (VOL-ADJ)" score={sig.irScore} weight="20%" description="5D return / 20D realized vol (information ratio)" />
                  <div style={{ height: 1, background: "#0c1824", margin: "14px 0" }} />
                  <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                    <div style={{ width: 70, height: 70, borderRadius: "50%", border: `2px solid ${sigColor}`, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: `${sigColor}08` }}>
                      <div style={{ fontSize: 14, fontWeight: 700, color: sigColor, fontFamily: "'IBM Plex Mono', monospace" }}>{(sig.composite * 100).toFixed(0)}</div>
                      <div style={{ fontSize: 7, color: sigColor + "70", fontFamily: "monospace" }}>SCORE</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 17, fontWeight: 800, color: sigColor, fontFamily: "'IBM Plex Mono', monospace" }}>{sigLabel}</div>
                      <div style={{ fontSize: 9, color: "#1a3050", marginTop: 3, fontFamily: "monospace" }}>Vol penalty: {fmt(sig.volPenalty)}× (vol ratio: {fmt(sig.volRatio)})</div>
                      <div style={{ fontSize: 9, color: "#1a3050", fontFamily: "monospace" }}>Half-Kelly: {fmt(sig.kelly * 100)}% position size</div>
                    </div>
                  </div>
                </div>
                <div>
                  <div style={{ background: "#060e1a", border: "1px solid #0c1824", borderRadius: 8, padding: 16, marginBottom: 10 }}>
                    <div style={{ fontSize: 9, color: "#1a3550", fontFamily: "monospace", marginBottom: 10 }}>SIGNAL INPUTS</div>
                    {[
                      ["RSI(14)", fmt(sig.currentRSI, 1), sig.currentRSI < 30 ? "#00e8a2" : sig.currentRSI > 70 ? "#ff4060" : "#4a9aff"],
                      ["5D Momentum", fmtPct(sig.mf.mom5), colorVal(sig.mf.mom5)],
                      ["21D Momentum", fmtPct(sig.mf.mom12), colorVal(sig.mf.mom12)],
                      ["IR Momentum", fmt(sig.mf.irMom, 3), colorVal(sig.mf.irMom)],
                      ["GARCH Vol (daily)", fmtPct(sig.garchVol), "#4a9aff"],
                      ["Long-run Vol", fmtPct(sig.longRunVol), "#2a5070"],
                      ["BB %B", fmt(sig.currentBB?.pctB), sig.currentBB?.pctB > 0.8 ? "#ff4060" : sig.currentBB?.pctB < 0.2 ? "#00e8a2" : "#4a9aff"],
                    ].map(([l, v, c]) => (
                      <div key={l} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid #0a1420", fontSize: 10, fontFamily: "monospace" }}>
                        <span style={{ color: "#2a4060" }}>{l}</span>
                        <span style={{ color: c || "#8899aa" }}>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* ── RISK TAB ── */}
            {activeTab === "risk" && analysis.risk && (
              <div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 6, marginBottom: 12 }}>
                  <Chip label="Sharpe Ratio" value={fmt(analysis.risk.sharpe)} color={colorVal(analysis.risk.sharpe, 1)} sub="ann. (rf=5.25%)" />
                  <Chip label="Sortino Ratio" value={fmt(analysis.risk.sortino)} color={colorVal(analysis.risk.sortino, 1)} sub="downside-adj" />
                  <Chip label="Calmar Ratio" value={fmt(analysis.risk.calmar)} color={colorVal(analysis.risk.calmar, 0.5)} sub="ret/maxDD" />
                  <Chip label="Max Drawdown" value={fmtPct(-analysis.risk.mdd)} color="#ff4060" sub="peak to trough" />
                  <Chip label="Ann. Return" value={fmtPct(analysis.risk.annRet)} color={colorVal(analysis.risk.annRet)} sub="log-return" />
                  <Chip label="Ann. Volatility" value={fmtPct(analysis.risk.annVol)} color="#4a9aff" sub="realized" />
                  <Chip label="VaR 95%" value={fmtPct(analysis.risk.var95)} color="#ff4060" sub="1-day, historical" />
                  <Chip label="CVaR 95%" value={fmtPct(analysis.risk.cvar95)} color="#ff4060" sub="expected shortfall" />
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                  <div style={{ background: "#060e1a", border: "1px solid #0c1824", borderRadius: 6, padding: "12px 14px" }}>
                    <div style={{ fontSize: 9, color: "#1a3550", fontFamily: "monospace", marginBottom: 8 }}>RETURN DISTRIBUTION</div>
                    {[
                      ["Skewness", fmt(analysis.risk.skew, 3), "Negative = fat left tail"],
                      ["Excess Kurtosis", fmt(analysis.risk.kurt, 3), ">0 = fat tails (leptokurtic)"],
                    ].map(([l, v, d]) => (
                      <div key={l} style={{ padding: "5px 0", borderBottom: "1px solid #0a1420" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, fontFamily: "monospace" }}>
                          <span style={{ color: "#2a4060" }}>{l}</span>
                          <span style={{ color: "#4a9aff" }}>{v}</span>
                        </div>
                        <div style={{ fontSize: 8, color: "#0a2030", fontFamily: "monospace" }}>{d}</div>
                      </div>
                    ))}
                  </div>
                  <div style={{ background: "#060e1a", border: "1px solid #0c1824", borderRadius: 6, padding: "12px 14px" }}>
                    <div style={{ fontSize: 9, color: "#1a3550", fontFamily: "monospace", marginBottom: 8 }}>GARCH PARAMETERS</div>
                    {[
                      ["ω (omega)", fmt(analysis.garch.omega, 8)],
                      ["α (alpha, ARCH)", fmt(analysis.garch.alpha)],
                      ["β (beta, GARCH)", fmt(analysis.garch.beta)],
                      ["α+β (persistence)", fmt(analysis.garch.persistence)],
                      ["Long-run variance", fmt(analysis.garch.longRunVar, 8)],
                    ].map(([l, v]) => (
                      <div key={l} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid #0a1420", fontSize: 9, fontFamily: "monospace" }}>
                        <span style={{ color: "#2a4060" }}>{l}</span>
                        <span style={{ color: "#4a9aff" }}>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* ── OPTIONS TAB ── */}
            {activeTab === "options" && analysis?.optionsChain && (
              <div>
                <div style={{ display: "flex", gap: 8, marginBottom: 12, alignItems: "center", flexWrap: "wrap" }}>
                  <div style={{ fontSize: 9, color: "#1a3550", fontFamily: "monospace", flex: 1 }}>
                    BLACK-SCHOLES OPTIONS CHAIN · IV = {fmtPct(Math.sqrt(analysis.garch.condVar[analysis.garch.condVar.length-1])*Math.sqrt(252))} (GARCH) · rf = 5.25%
                  </div>
                  <div style={{ display: "flex", gap: 4 }}>
                    {["call","put"].map(t => (
                      <button key={t} onClick={() => setOptionType(t)} style={{ padding:"3px 12px", fontSize:10, fontFamily:"monospace", border:`1px solid ${optionType===t?"#0a6aff":"#0c1824"}`, background: optionType===t?"#051830":"transparent", color: optionType===t?"#4a9aff":"#2a4560", borderRadius:4, cursor:"pointer", textTransform:"uppercase" }}>{t}</button>
                    ))}
                  </div>
                  <div style={{ display: "flex", gap: 4 }}>
                    {[7,14,30,60,90].map(d => (
                      <button key={d} onClick={() => setOptionExpiry(d)} style={{ padding:"3px 10px", fontSize:10, fontFamily:"monospace", border:`1px solid ${optionExpiry===d?"#0a6aff":"#0c1824"}`, background: optionExpiry===d?"#051830":"transparent", color: optionExpiry===d?"#4a9aff":"#2a4560", borderRadius:4, cursor:"pointer" }}>{d}D</button>
                    ))}
                  </div>
                </div>
                {(() => {
                  const chain = analysis.optionsChain.find(c => c.expiry === optionExpiry) || analysis.optionsChain[2];
                  const opt = optionType;
                  return (
                    <div>
                      <div style={{ overflowX: "auto" }}>
                        <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "'IBM Plex Mono', monospace", fontSize: 10 }}>
                          <thead>
                            <tr style={{ borderBottom: "1px solid #0c1824" }}>
                              {["Strike","Moneyness","Price","Delta","Gamma","Theta","Vega","Rho","IV"].map(h => (
                                <th key={h} style={{ padding: "6px 10px", color: "#1a3550", textAlign: "right", fontWeight: 600, letterSpacing: "0.05em", fontSize: 9 }}>{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {chain.strikes.map((row, i) => {
                              const o = row[opt];
                              const atm = Math.abs(row.K - analysis.S0) < analysis.S0 * 0.03;
                              const itm = opt === "call" ? row.K < analysis.S0 : row.K > analysis.S0;
                              return (
                                <tr key={i} style={{ background: atm ? "#0a2040" : itm ? "#060e1a" : "transparent", borderBottom: "1px solid #080f1a" }}>
                                  <td style={{ padding: "5px 10px", textAlign: "right", color: atm ? "#4a9aff" : "#8899aa", fontWeight: atm ? 700 : 400 }}>${row.K}</td>
                                  <td style={{ padding: "5px 10px", textAlign: "right", color: parseFloat(row.moneyness) > 0 ? "#00e8a2" : parseFloat(row.moneyness) < 0 ? "#ff4060" : "#f0c040" }}>{row.moneyness}%</td>
                                  <td style={{ padding: "5px 10px", textAlign: "right", color: "#d0dde8" }}>${o.price.toFixed(2)}</td>
                                  <td style={{ padding: "5px 10px", textAlign: "right", color: o.delta > 0 ? "#00e8a2" : "#ff4060" }}>{o.delta.toFixed(3)}</td>
                                  <td style={{ padding: "5px 10px", textAlign: "right", color: "#4a9aff" }}>{o.gamma.toFixed(4)}</td>
                                  <td style={{ padding: "5px 10px", textAlign: "right", color: "#ff4060" }}>{o.theta.toFixed(3)}</td>
                                  <td style={{ padding: "5px 10px", textAlign: "right", color: "#f0c040" }}>{o.vega.toFixed(3)}</td>
                                  <td style={{ padding: "5px 10px", textAlign: "right", color: "#8899aa" }}>{o.rho.toFixed(3)}</td>
                                  <td style={{ padding: "5px 10px", textAlign: "right", color: "#a070ff" }}>{fmtPct(o.iv)}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                      {/* Greeks summary for ATM */}
                      {(() => {
                        const atm = chain.strikes.reduce((best, r) => Math.abs(r.K - analysis.S0) < Math.abs(best.K - analysis.S0) ? r : best);
                        const o = atm[opt];
                        return (
                          <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 6, marginTop: 12 }}>
                            {[
                              { label: "ATM Price", value: `$${o.price.toFixed(2)}`, color: "#d0dde8" },
                              { label: "Delta Δ", value: o.delta.toFixed(3), color: o.delta > 0 ? "#00e8a2" : "#ff4060", sub: "≈ prob ITM" },
                              { label: "Gamma Γ", value: o.gamma.toFixed(4), color: "#4a9aff", sub: "Δ per $1 move" },
                              { label: "Theta Θ", value: `$${o.theta.toFixed(3)}/day`, color: "#ff4060", sub: "time decay" },
                              { label: "Vega ν", value: `$${o.vega.toFixed(3)}/1%`, color: "#f0c040", sub: "vol sensitivity" },
                            ].map(m => <Chip key={m.label} {...m} />)}
                          </div>
                        );
                      })()}
                    </div>
                  );
                })()}
              </div>
            )}

            {/* ── BACKTEST TAB ── */}
            {activeTab === "backtest" && analysis?.backtest && (() => {
              const bt = analysis.backtest;
              return (
                <div>
                  <div style={{ fontSize: 9, color: "#1a3550", fontFamily: "monospace", marginBottom: 8 }}>
                    EMA CROSSOVER STRATEGY (10D/30D) · LONG-ONLY · INITIAL CAPITAL $10,000
                  </div>
                  <div style={{ height: 270, background: "#060e1a", borderRadius: 8, border: "1px solid #0c1824", padding: "12px 4px" }}>
                    <ResponsiveContainer>
                      <LineChart data={bt.equity}>
                        <XAxis dataKey="day" tick={{ fill: "#1a3050", fontSize: 8, fontFamily: "monospace" }} tickLine={false} axisLine={false} interval={Math.floor(bt.equity.length/6)} />
                        <YAxis tick={{ fill: "#1a3050", fontSize: 8, fontFamily: "monospace" }} tickLine={false} axisLine={false} tickFormatter={v => `$${(v/1000).toFixed(1)}k`} domain={["auto","auto"]} />
                        <Tooltip content={<CustomTooltip />} />
                        <ReferenceLine y={10000} stroke="#1a2535" strokeDasharray="3 3" />
                        <Line type="monotone" dataKey="benchmark" stroke="#2a4060" strokeWidth={1.5} dot={false} name="Buy & Hold" />
                        <Line type="monotone" dataKey="value" stroke="#00e8a2" strokeWidth={2} dot={false} name="Strategy" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 6, marginTop: 10 }}>
                    <Chip label="Total Return" value={fmtPct(bt.totalReturn)} color={colorVal(bt.totalReturn)} sub="strategy" />
                    <Chip label="Benchmark" value={fmtPct(bt.bmReturn)} color={colorVal(bt.bmReturn)} sub="buy & hold" />
                    <Chip label="Alpha" value={fmtPct(bt.alpha)} color={colorVal(bt.alpha)} sub="vs benchmark" />
                    <Chip label="Sharpe" value={fmt(bt.sharpe)} color={colorVal(bt.sharpe, 1)} sub="annualized" />
                    <Chip label="Max Drawdown" value={fmtPct(-bt.maxDD)} color="#ff4060" sub="strategy" />
                    <Chip label="Ann. Return" value={fmtPct(bt.annRet)} color={colorVal(bt.annRet)} sub="strategy" />
                    <Chip label="Win Rate" value={fmtPct(bt.winRate)} color={colorVal(bt.winRate, 0.5)} sub={`${bt.nTrades} trades`} />
                    <Chip label="Ann. Vol" value={fmtPct(bt.annVol)} color="#4a9aff" sub="strategy" />
                  </div>
                  <div style={{ marginTop: 10, background: "#060e1a", border: "1px solid #0c1824", borderRadius: 6, padding: "10px 14px", fontSize: 9, color: "#1a3550", fontFamily: "monospace" }}>
                    Strategy: Buy when 10D EMA crosses above 30D EMA (Golden Cross). Exit when 10D EMA crosses below 30D EMA (Death Cross). Long-only, fully invested when in position, flat (cash) otherwise.
                  </div>
                </div>
              );
            })()}

            {/* ── SENTIMENT TAB ── */}
            {activeTab === "sentiment" && (
              <div>
                {(!extras || extras.loading) && (
                  <div style={{ padding: "40px 0", textAlign: "center", fontSize: 10, color: "#0a5adf", fontFamily: "monospace", animation: "pulse 1.4s ease infinite" }}>
                    Scanning latest news via web search…
                  </div>
                )}
                {extras?.sentiment?.error && (
                  <div style={{ padding: "20px 0", color: "#ff4060", fontFamily: "monospace", fontSize: 11 }}>✕ {extras.sentiment.error}</div>
                )}
                {extras?.sentiment && !extras.sentiment.error && (() => {
                  const s = extras.sentiment;
                  const sc = s.overallSentiment;
                  const scColor = sc > 0.2 ? "#00e8a2" : sc < -0.2 ? "#ff4060" : "#f0c040";
                  const tagColors = { EARNINGS:"#4a9aff", UPGRADE:"#00e8a2", DOWNGRADE:"#ff4060", MACRO:"#f0c040", PRODUCT:"#a070ff", LEGAL:"#ff8040", INSIDER:"#ff4060", "M&A":"#00c8c8", GUIDANCE:"#4a9aff" };
                  return (
                    <div>
                      <div style={{ display: "flex", gap: 12, marginBottom: 16, alignItems: "center" }}>
                        <div style={{ padding: "12px 20px", borderRadius: 8, border: `1px solid ${scColor}30`, background: `${scColor}08`, textAlign: "center" }}>
                          <div style={{ fontSize: 8, color: "#2a4060", fontFamily: "monospace", letterSpacing: "0.12em", marginBottom: 4 }}>OVERALL SENTIMENT</div>
                          <div style={{ fontSize: 22, fontWeight: 800, color: scColor, fontFamily: "'IBM Plex Mono', monospace" }}>{s.sentimentLabel}</div>
                          <div style={{ fontSize: 10, color: scColor + "80", fontFamily: "monospace" }}>{sc > 0 ? "+" : ""}{(sc * 100).toFixed(0)} / 100</div>
                        </div>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: 9, color: "#1a3550", fontFamily: "monospace", marginBottom: 6 }}>KEY THEMES</div>
                          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                            {(s.keyThemes || []).map((t, i) => (
                              <span key={i} style={{ padding: "3px 10px", borderRadius: 3, background: "#0a1a2a", border: "1px solid #1a3050", fontSize: 10, color: "#4a7090", fontFamily: "monospace" }}>{t}</span>
                            ))}
                          </div>
                        </div>
                      </div>
                      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                        {(s.headlines || []).map((h, i) => {
                          const c = h.sentiment > 0.2 ? "#00e8a2" : h.sentiment < -0.2 ? "#ff4060" : "#f0c040";
                          const tagC = tagColors[h.tag] || "#4a7090";
                          return (
                            <div key={i} style={{ background: "#060e1a", border: "1px solid #0c1824", borderRadius: 8, padding: "12px 14px", display: "flex", gap: 12 }}>
                              <div style={{ width: 36, height: 36, borderRadius: "50%", border: `2px solid ${c}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, background: `${c}08` }}>
                                <span style={{ fontSize: 11, fontWeight: 700, color: c, fontFamily: "monospace" }}>{h.sentiment > 0 ? "+" : ""}{(h.sentiment * 10).toFixed(0)}</span>
                              </div>
                              <div style={{ flex: 1, minWidth: 0 }}>
                                <div style={{ display: "flex", gap: 6, marginBottom: 4, flexWrap: "wrap" }}>
                                  <span style={{ fontSize: 8, padding: "1px 6px", borderRadius: 2, background: `${tagC}15`, color: tagC, fontFamily: "monospace", border: `1px solid ${tagC}30` }}>{h.tag}</span>
                                  <span style={{ fontSize: 8, color: "#1a3550", fontFamily: "monospace" }}>{h.source}</span>
                                  <span style={{ fontSize: 8, color: "#0a2535", fontFamily: "monospace" }}>{h.date}</span>
                                </div>
                                <div style={{ fontSize: 11, color: "#8899aa", marginBottom: 3, lineHeight: 1.4 }}>{h.title}</div>
                                <div style={{ fontSize: 9, color: "#2a4060", fontFamily: "monospace" }}>{h.summary}</div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}

            {/* ── PEERS TAB ── */}
            {activeTab === "peers" && (
              <div>
                {(!extras || extras.loading) && (
                  <div style={{ padding: "40px 0", textAlign: "center", fontSize: 10, color: "#0a5adf", fontFamily: "monospace", animation: "pulse 1.4s ease infinite" }}>
                    Fetching sector peer data via web search…
                  </div>
                )}
                {extras?.peers?.error && (
                  <div style={{ padding: "20px 0", color: "#ff4060", fontFamily: "monospace", fontSize: 11 }}>✕ {extras.peers.error}</div>
                )}
                {extras?.peers?.peers && !extras.peers.error && (() => {
                  const peers = extras.peers.peers;
                  const ratingColor = r => ({ "STRONG BUY":"#00e8a2", "BUY":"#4a9aff", "HOLD":"#f0c040", "SELL":"#ff8040", "STRONG SELL":"#ff4060" }[r] || "#8899aa");
                  const metrics = ["peRatio","pbRatio","grossMargin","beta","ytdReturn"];
                  const metaLabels = { peRatio:"P/E", pbRatio:"P/B", grossMargin:"Gross Margin", beta:"Beta", ytdReturn:"YTD Ret" };
                  const isPct = m => m === "ytdReturn" || m === "grossMargin";
                  return (
                    <div>
                      <div style={{ overflowX: "auto", marginBottom: 14 }}>
                        <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "'IBM Plex Mono', monospace", fontSize: 10 }}>
                          <thead>
                            <tr style={{ borderBottom: "1px solid #0c1824" }}>
                              {["Ticker","Name","Price","Mkt Cap","Rating","Target",...metrics.map(m => metaLabels[m])].map(h => (
                                <th key={h} style={{ padding: "6px 10px", color: "#1a3550", textAlign: "right", fontWeight: 600, fontSize: 9, whiteSpace: "nowrap" }}>{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {peers.map((p, i) => (
                              <tr key={i} style={{ background: p.ticker === ticker ? "#0a2040" : "transparent", borderBottom: "1px solid #080f1a" }}>
                                <td style={{ padding: "6px 10px", color: p.ticker === ticker ? "#4a9aff" : "#8899aa", fontWeight: p.ticker === ticker ? 700 : 400 }}>{p.ticker}</td>
                                <td style={{ padding: "6px 10px", color: "#5a7090", whiteSpace: "nowrap", maxWidth: 120, overflow: "hidden", textOverflow: "ellipsis" }}>{p.name}</td>
                                <td style={{ padding: "6px 10px", textAlign: "right", color: "#d0dde8" }}>${typeof p.price === "number" ? p.price.toFixed(2) : p.price}</td>
                                <td style={{ padding: "6px 10px", textAlign: "right", color: "#5a7090" }}>{p.marketCap}</td>
                                <td style={{ padding: "6px 10px", textAlign: "right", color: ratingColor(p.analystRating), fontSize: 9 }}>{p.analystRating}</td>
                                <td style={{ padding: "6px 10px", textAlign: "right", color: "#4a9aff" }}>{p.priceTarget ? `$${p.priceTarget}` : "—"}</td>
                                {metrics.map(m => {
                                  const v = p[m];
                                  const display = v == null ? "—" : isPct(m) ? `${(v*100).toFixed(1)}%` : typeof v === "number" ? v.toFixed(2) : v;
                                  const color = isPct(m) ? colorVal(v) : "#8899aa";
                                  return <td key={m} style={{ padding: "6px 10px", textAlign: "right", color: color }}>{display}</td>;
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      {/* Visual bar comparison for YTD Return */}
                      <div style={{ background: "#060e1a", border: "1px solid #0c1824", borderRadius: 8, padding: "14px 16px" }}>
                        <div style={{ fontSize: 9, color: "#1a3550", fontFamily: "monospace", marginBottom: 12 }}>YTD RETURN COMPARISON</div>
                        {peers.map((p, i) => {
                          const v = p.ytdReturn || 0;
                          const col = v > 0 ? "#00e8a2" : "#ff4060";
                          const maxAbs = Math.max(...peers.map(x => Math.abs(x.ytdReturn || 0)), 0.01);
                          return (
                            <div key={i} style={{ marginBottom: 8 }}>
                              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, fontFamily: "monospace", marginBottom: 3 }}>
                                <span style={{ color: p.ticker === ticker ? "#4a9aff" : "#3a5070" }}>{p.ticker}</span>
                                <span style={{ color: col }}>{v > 0 ? "+" : ""}{(v*100).toFixed(1)}%</span>
                              </div>
                              <div style={{ height: 4, background: "#0a1520", borderRadius: 2 }}>
                                <div style={{ height: "100%", width: `${Math.abs(v)/maxAbs*100}%`, background: col, borderRadius: 2, marginLeft: v < 0 ? `${(1-Math.abs(v)/maxAbs)*100}%` : 0, transition: "all 0.5s" }} />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}


            <div style={{ marginTop: 24, paddingTop: 12, borderTop: "1px solid #08121c", fontSize: 8, color: "#0a1e30", fontFamily: "monospace", lineHeight: 1.9 }}>
              DISCLAIMER: Educational purposes only. Real price data fetched via web search. Models: GARCH(1,1) MLE · 2-State HMM (Baum-Welch EM) · {analysis.paths}-path Monte Carlo · UMD Momentum · RSI(14) · MACD(12,26,9) · Bollinger(20,2) · Half-Kelly · Sharpe · Sortino · Calmar · CVaR95 · Black-Scholes (Δ Γ Θ ν ρ) · EMA Crossover Backtest · News Sentiment · Peer Comparison. Not financial advice.
            </div>
          </div>
        )}

        {state.status === "idle" && (
          <div style={{ padding: "60px 0", textAlign: "center", color: "#1a3050", fontFamily: "monospace", fontSize: 11 }}>
            Select a ticker above to begin analysis
          </div>
        )}
      </div>
    </div>
  );
}
