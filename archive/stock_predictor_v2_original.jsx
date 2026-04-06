import { useState, useEffect, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, AreaChart, Area, ComposedChart, Bar } from "recharts";

// ═══════════════════════════════════════════════════════════════
//  MATH
// ═══════════════════════════════════════════════════════════════
const mean = a => a.reduce((s,v)=>s+v,0)/a.length;
const sumArr = a => a.reduce((s,v)=>s+v,0);
const stdDev = a => { const m=mean(a); return Math.sqrt(a.reduce((s,v)=>s+(v-m)**2,0)/a.length); };
const calcVar = a => { const m=mean(a); return a.reduce((s,v)=>s+(v-m)**2,0)/a.length; };
const clamp = (v,lo,hi) => Math.max(lo,Math.min(hi,v));
const logRet = prices => prices.slice(1).map((p,i)=>Math.log(p/prices[i]));
const simRet = prices => prices.slice(1).map((p,i)=>(p-prices[i])/prices[i]);
const ema = (arr,n) => arr.reduce((acc,v,i)=>{ const k=2/(n+1); return i===0?[v]:[...acc,v*k+acc[i-1]*(1-k)]; },[]);
function boxMuller(){ let u=0,v=0; while(!u)u=Math.random(); while(!v)v=Math.random(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }
function seededRand(seed){ let s=seed; return ()=>{ s=(s*1664525+1013904223)&0xffffffff; return (s>>>0)/0xffffffff; }; }
function seededBM(rand){ let u=0,v=0; do{u=rand();}while(!u); do{v=rand();}while(!v); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }

// ═══════════════════════════════════════════════════════════════
//  GARCH(1,1)
// ═══════════════════════════════════════════════════════════════
function fitGARCH(returns){
  if(returns.length<5) return {omega:1e-5,alpha:0.08,beta:0.88,condVar:[1e-4],longRunVar:1e-4,persistence:0.96};
  const r2=returns.map(r=>r*r), uv=Math.max(calcVar(returns),1e-8);
  let omega=uv*0.05, alpha=0.08, beta=0.88;
  for(let iter=0;iter<60;iter++){
    let h=[uv], logL=0;
    for(let t=1;t<returns.length;t++){ const ht=Math.max(omega+alpha*r2[t-1]+beta*h[t-1],1e-10); h.push(ht); logL+=-0.5*(Math.log(ht)+r2[t]/ht); }
    let best=logL, bo=omega, ba=alpha, bb=beta;
    for(const [wo,a,b] of [[omega*1.02,alpha,beta],[omega*0.98,alpha,beta],[omega,alpha*1.05,beta],[omega,alpha*0.95,beta],[omega,alpha,beta*1.01],[omega,alpha,beta*0.99]]){
      if(a+b>=0.9999||a<0||b<0||wo<=0) continue;
      let h2=[uv],l2=0;
      for(let t=1;t<returns.length;t++){ const ht=Math.max(wo+a*r2[t-1]+b*h2[t-1],1e-10); h2.push(ht); l2+=-0.5*(Math.log(ht)+r2[t]/ht); }
      if(l2>best){best=l2;bo=wo;ba=a;bb=b;}
    }
    omega=bo;alpha=ba;beta=bb;
  }
  const condVar=[uv];
  for(let t=1;t<returns.length;t++) condVar.push(Math.max(omega+alpha*r2[t-1]+beta*condVar[t-1],1e-10));
  const denom=1-alpha-beta; const longRunVar=denom>0.001?omega/denom:uv;
  return {omega,alpha,beta,condVar,longRunVar,persistence:alpha+beta};
}

// ═══════════════════════════════════════════════════════════════
//  HMM (2-state Baum-Welch)
// ═══════════════════════════════════════════════════════════════
function fitHMM(returns,nIter=25){
  const fallback={states:[],stateProbs:[[0.6,0.4]],muS:[0.0005,-0.0005],sigS:[0.01,0.02],A:[[0.95,0.05],[0.10,0.90]],currentState:0,bullProb:0.6};
  if(returns.length<10) return fallback;
  const n=returns.length;
  const evens=returns.filter((_,i)=>i%2===0), odds=returns.filter((_,i)=>i%2!==0);
  let muS=[mean(evens)||0.0005, mean(odds)||-0.0005];
  let sigS=[Math.max(stdDev(evens),0.001), Math.max(stdDev(odds),0.001)];
  let A=[[0.95,0.05],[0.10,0.90]], pi=[0.6,0.4];
  const g=(x,mu,sig)=>(1/Math.max(sig*Math.sqrt(2*Math.PI),1e-10))*Math.exp(-0.5*((x-mu)/Math.max(sig,1e-10))**2);
  for(let iter=0;iter<nIter;iter++){
    const af=Array.from({length:n},()=>[0,0]), sc=new Array(n).fill(0);
    for(let s=0;s<2;s++) af[0][s]=pi[s]*g(returns[0],muS[s],sigS[s]);
    sc[0]=Math.max(sumArr(af[0]),1e-30); for(let s=0;s<2;s++) af[0][s]/=sc[0];
    for(let t=1;t<n;t++){
      for(let j=0;j<2;j++) af[t][j]=sumArr([0,1].map(i=>af[t-1][i]*A[i][j]))*g(returns[t],muS[j],sigS[j]);
      sc[t]=Math.max(sumArr(af[t]),1e-30); for(let j=0;j<2;j++) af[t][j]/=sc[t];
    }
    const bb=Array.from({length:n},()=>[1,1]);
    for(let t=n-2;t>=0;t--){
      for(let i=0;i<2;i++) bb[t][i]=sumArr([0,1].map(j=>A[i][j]*g(returns[t+1],muS[j],sigS[j])*bb[t+1][j]));
      const sc2=Math.max(sumArr(bb[t]),1e-30); for(let i=0;i<2;i++) bb[t][i]/=sc2;
    }
    const gamma=af.map((a,t)=>{ const gv=[a[0]*bb[t][0],a[1]*bb[t][1]]; const sv=Math.max(sumArr(gv),1e-30); return gv.map(x=>x/sv); });
    const xi=Array.from({length:n-1},(_,t)=>{
      const mat=[[0,0],[0,0]]; let tot=0;
      for(let i=0;i<2;i++) for(let j=0;j<2;j++){mat[i][j]=af[t][i]*A[i][j]*g(returns[t+1],muS[j],sigS[j])*bb[t+1][j];tot+=mat[i][j];}
      const tt=Math.max(tot,1e-30); for(let i=0;i<2;i++) for(let j=0;j<2;j++) mat[i][j]/=tt; return mat;
    });
    pi=[gamma[0][0],gamma[0][1]];
    for(let i=0;i<2;i++){ const gs=Math.max(sumArr(xi.map(x=>sumArr(x[i]))),1e-30); A[i]=[sumArr(xi.map(x=>x[i][0]))/gs,sumArr(xi.map(x=>x[i][1]))/gs]; }
    for(let j=0;j<2;j++){
      const gj=gamma.map(gv=>gv[j]), gs=Math.max(sumArr(gj),1e-30);
      muS[j]=sumArr(gj.map((gv,t)=>gv*returns[t]))/gs;
      sigS[j]=Math.max(Math.sqrt(sumArr(gj.map((gv,t)=>gv*(returns[t]-muS[j])**2))/gs),0.001);
    }
  }
  const sp=[], fw=Array.from({length:n},()=>[0,0]);
  const sc0=[0,1].map(s=>pi[s]*g(returns[0],muS[s],sigS[s]));
  const sc0t=Math.max(sumArr(sc0),1e-30);
  for(let s=0;s<2;s++) fw[0][s]=sc0[s]/sc0t; sp.push([...fw[0]]);
  for(let t=1;t<n;t++){
    const obs=[0,1].map(j=>sumArr([0,1].map(i=>fw[t-1][i]*A[i][j]))*g(returns[t],muS[j],sigS[j]));
    const sc=Math.max(sumArr(obs),1e-30); for(let j=0;j<2;j++) fw[t][j]=obs[j]/sc; sp.push([...fw[t]]);
  }
  const states=sp.map(p=>p[0]>p[1]?0:1), cur=states[states.length-1], bullP=sp[sp.length-1][0];
  const swap=muS[0]<muS[1];
  return {states:swap?states.map(s=>1-s):states,stateProbs:swap?sp.map(p=>[p[1],p[0]]):sp,
    muS:swap?[muS[1],muS[0]]:muS,sigS:swap?[sigS[1],sigS[0]]:sigS,
    A:swap?[[A[1][1],A[1][0]],[A[0][1],A[0][0]]]:A,
    currentState:swap?1-cur:cur,bullProb:swap?1-bullP:bullP};
}

// ═══════════════════════════════════════════════════════════════
//  INDICATORS
// ═══════════════════════════════════════════════════════════════
function computeRSI(prices,period=14){
  const rets=simRet(prices); const out=[];
  for(let i=period;i<=rets.length;i++){
    const w=rets.slice(i-period,i);
    const ag=mean(w.filter(r=>r>0).concat(0.0001)), al=mean(w.filter(r=>r<0).map(Math.abs).concat(0.0001));
    out.push(100-100/(1+ag/al));
  }
  return out;
}
function computeMACD(prices,fast=12,slow=26,sig=9){
  const fe=ema(prices,fast), se=ema(prices,slow);
  const macdLine=fe.slice(slow-fast).map((v,i)=>v-se[i+slow-fast]);
  const sigLine=ema(macdLine,sig);
  return {macdLine,sigLine,hist:macdLine.slice(sig-1).map((v,i)=>v-sigLine[i])};
}
function bollingerBands(prices,period=20,k=2){
  return prices.slice(period-1).map((_,i)=>{
    const w=prices.slice(i,i+period), m=mean(w), s=Math.max(stdDev(w),0.0001);
    return {mid:m,upper:m+k*s,lower:m-k*s,pctB:(prices[i+period-1]-(m-k*s))/(2*k*s)};
  });
}
function atrFn(prices,period=14){
  const trs=prices.slice(1).map((p,i)=>Math.abs(p-prices[i])); const out=[];
  for(let i=period;i<=trs.length;i++) out.push(mean(trs.slice(i-period,i)));
  return out;
}
function riskMetrics(returns,prices,rf=0.0525/252){
  if(!returns.length||returns.length<2) return {sharpe:0,sortino:0,calmar:0,mdd:0,skew:0,kurt:0,var95:0,cvar95:0,annVol:0,annRet:0};
  const excess=returns.map(r=>r-rf), sv=Math.max(stdDev(returns),1e-10);
  const sharpe=mean(excess)/sv*Math.sqrt(252);
  const dn=returns.filter(r=>r<rf); const sortino=mean(excess)/Math.max(stdDev(dn.length?dn:[0]),1e-10)*Math.sqrt(252);
  let peak=prices[0],mdd=0; for(const p of prices){if(p>peak)peak=p;mdd=Math.max(mdd,(peak-p)/Math.max(peak,1e-10));}
  const annRet=mean(returns)*252, calmar=mdd>0.001?annRet/mdd:0;
  const m=mean(returns),s=Math.max(stdDev(returns),1e-10);
  const skew=mean(returns.map(r=>((r-m)/s)**3)), kurt=mean(returns.map(r=>((r-m)/s)**4))-3;
  const sorted=[...returns].sort((a,b)=>a-b), tn=Math.max(1,Math.floor(returns.length*0.05));
  return {sharpe,sortino,calmar,mdd,skew,kurt,var95:sorted[tn]||0,cvar95:mean(sorted.slice(0,tn)),annVol:sv*Math.sqrt(252),annRet};
}
function garchMC(S0,returns,garch,hmm,paths=300,horizon=30){
  const {omega,alpha,beta}=garch;
  const h0=Math.max(garch.condVar[garch.condVar.length-1],1e-8);
  const mu=(hmm.muS&&hmm.currentState!=null)?hmm.muS[hmm.currentState]:mean(returns);
  return Array.from({length:paths},()=>{
    const path=[S0]; let ht=h0, pe=boxMuller()*Math.sqrt(h0);
    for(let t=1;t<=horizon;t++){
      ht=Math.max(omega+alpha*pe**2+beta*ht,1e-10);
      const eps=boxMuller()*Math.sqrt(ht); pe=eps;
      path.push(Math.max(path[path.length-1]*Math.exp(mu-0.5*ht+eps),0.01));
    }
    return path;
  });
}
function calibratedSignal(returns,prices,hmm,garch,rsiVals,bbands){
  const empty={composite:0,trendScore:0,reversionScore:0,regimeScore:0,irScore:0,volPenalty:1,bullProb:0.5,currentRegime:0,garchVol:0.01,longRunVol:0.01,volRatio:1,persistence:0.9,kelly:0,mf:{mom1:0,mom5:0,mom12:0,vol20:0.01,irMom:0},currentRSI:50,currentBB:null};
  if(!returns.length||returns.length<5||!rsiVals.length) return empty;
  const safeSlice=(arr,n)=>arr.slice(-Math.min(n,arr.length));
  const mom1=mean(safeSlice(returns,5)), mom5=mean(safeSlice(returns,21)), mom12=mean(safeSlice(returns,Math.min(63,returns.length)));
  const vol20=Math.max(stdDev(safeSlice(returns,20)),0.001), irMom=mom5/vol20;
  const mf={mom1,mom5,mom12,vol20,irMom};
  const currentRSI=rsiVals[rsiVals.length-1]??50, currentBB=bbands.length?bbands[bbands.length-1]:null;
  const bullProb=hmm.bullProb??0.5, currentRegime=hmm.currentState??0;
  const garchVol=Math.sqrt(Math.max(garch.condVar[garch.condVar.length-1],1e-8));
  const longRunVol=Math.sqrt(Math.max(garch.longRunVar,1e-8));
  const volRatio=garchVol/Math.max(longRunVol,1e-10);
  const umd=mom12-mom1, trendScore=clamp(umd*300,-1,1);
  const rsiScore=currentRSI<30?1:currentRSI>70?-1:(50-currentRSI)/50;
  const bbScore=currentBB?clamp(1-2*currentBB.pctB,-1,1):0;
  const reversionScore=0.6*rsiScore+0.4*bbScore;
  const regimeScore=currentRegime===0?bullProb:-(1-bullProb);
  const irScore=clamp(irMom*20,-1,1);
  const volPenalty=volRatio>1.5?0.4:volRatio>1.2?0.65:1.0;
  const composite=clamp((0.35*trendScore+0.20*reversionScore+0.25*regimeScore+0.20*irScore)*volPenalty,-1,1);
  const mu2=mean(returns)*252, sig2=Math.max(calcVar(returns)*252,1e-10);
  const kelly=clamp(mu2/sig2*0.5,-1,2);
  return {composite,trendScore,reversionScore,regimeScore,irScore,volPenalty,bullProb,currentRegime,garchVol,longRunVol,volRatio,persistence:garch.persistence,kelly,mf,currentRSI,currentBB};
}

// ═══════════════════════════════════════════════════════════════
//  BLACK-SCHOLES
// ═══════════════════════════════════════════════════════════════
function normCDF(x){
  const a1=0.254829592,a2=-0.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=0.3275911;
  const sign=x<0?-1:1; x=Math.abs(x)/Math.sqrt(2);
  const t=1/(1+p*x),y=1-(((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*Math.exp(-x*x);
  return 0.5*(1+sign*y);
}
function normPDF(x){return Math.exp(-0.5*x*x)/Math.sqrt(2*Math.PI);}
function bs(S,K,T,r,sigma,type="call"){
  if(T<=0){const iv=type==="call"?Math.max(S-K,0):Math.max(K-S,0);return {price:iv,delta:0,gamma:0,theta:0,vega:0,rho:0};}
  const d1=(Math.log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*Math.sqrt(T));
  const d2=d1-sigma*Math.sqrt(T),df=Math.exp(-r*T);
  const price=type==="call"?S*normCDF(d1)-K*df*normCDF(d2):K*df*normCDF(-d2)-S*normCDF(-d1);
  const delta=type==="call"?normCDF(d1):normCDF(d1)-1;
  const gamma=normPDF(d1)/(S*sigma*Math.sqrt(T));
  const vega=S*normPDF(d1)*Math.sqrt(T)/100;
  const theta=type==="call"?(-S*normPDF(d1)*sigma/(2*Math.sqrt(T))-r*K*df*normCDF(d2))/365:(-S*normPDF(d1)*sigma/(2*Math.sqrt(T))+r*K*df*normCDF(-d2))/365;
  const rho=type==="call"?K*T*df*normCDF(d2)/100:-K*T*df*normCDF(-d2)/100;
  return {price,delta,gamma,theta,vega,rho};
}
function buildChain(S,sigma,r=0.0525,expiries=[7,14,30,60,90]){
  const strikes=[-0.15,-0.10,-0.05,-0.02,0,0.02,0.05,0.10,0.15].map(pct=>Math.round(S*(1+pct)/5)*5||1);
  return expiries.map(days=>({expiry:days,strikes:strikes.map(K=>({K,call:bs(S,K,days/365,r,sigma,"call"),put:bs(S,K,days/365,r,sigma,"put"),moneyness:((S-K)/K*100).toFixed(1)}))}));
}

// ═══════════════════════════════════════════════════════════════
//  BACKTEST
// ═══════════════════════════════════════════════════════════════
function runBacktest(prices,returns){
  const empty={equity:[{day:0,value:10000,benchmark:10000}],trades:[],totalReturn:0,bmReturn:0,sharpe:0,maxDD:0,winRate:0,alpha:0,annRet:0,annVol:0,nTrades:0};
  if(prices.length<35) return empty;
  const e10=ema(prices,10),e30=ema(prices,30);
  const equity=[10000],bench=[10000],pnl=[],trades=[];
  let pos=0,entryPrice=0,peak=10000,maxDD=0;
  for(let i=30;i<prices.length-1;i++){
    const sig=e10[i]>e30[i]?1:0,ret=returns[i]||0;
    if(sig!==pos){
      if(pos!==0&&trades.length) trades[trades.length-1].pnl=pos*(prices[i]-entryPrice);
      if(sig!==0){trades.push({type:"BUY",price:prices[i],idx:i,pnl:null});entryPrice=prices[i];}
      pos=sig;
    }
    const newEq=equity[equity.length-1]*(1+pos*ret);
    equity.push(newEq);bench.push(bench[bench.length-1]*(1+ret));pnl.push(pos*ret);
    peak=Math.max(peak,newEq);maxDD=Math.max(maxDD,(peak-newEq)/Math.max(peak,1));
  }
  const tot=(equity[equity.length-1]-10000)/10000,bm=(bench[bench.length-1]-10000)/10000;
  const annRet=pnl.length?mean(pnl)*252:0,annVol=pnl.length?stdDev(pnl)*Math.sqrt(252):0;
  const closed=trades.filter(t=>t.pnl!=null);
  return {equity:equity.map((v,i)=>({day:i,value:+v.toFixed(2),benchmark:+(bench[i]||10000).toFixed(2)})),trades,totalReturn:tot,bmReturn:bm,sharpe:annVol>0.001?annRet/annVol:0,maxDD,winRate:closed.length?closed.filter(t=>t.pnl>0).length/closed.length:0,alpha:tot-bm,annRet,annVol,nTrades:closed.length};
}

// ═══════════════════════════════════════════════════════════════
//  SIMULATE HISTORY
// ═══════════════════════════════════════════════════════════════
function simulateHistory(currentPrice,annMu,annSigma,days=90,seed=42){
  const rand=seededRand(seed),dt=1/252;
  const path=[currentPrice];
  for(let i=0;i<days-1;i++){
    const z=seededBM(rand);
    path.unshift(Math.max(path[0]/Math.exp((annMu-0.5*annSigma**2)*dt+annSigma*Math.sqrt(dt)*z),0.01));
  }
  const MO=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const allDates=[],today=new Date();
  for(let i=0;allDates.length<days;i++){
    const d=new Date(today);d.setDate(today.getDate()-i);
    if(d.getDay()!==0&&d.getDay()!==6) allDates.push(`${MO[d.getMonth()]} ${d.getDate()}`);
  }
  return {prices:path,dates:allDates.reverse()};
}

// ═══════════════════════════════════════════════════════════════
//  METADATA
// ═══════════════════════════════════════════════════════════════
const STOCK_META={
  AAPL:{name:"Apple Inc.",       sector:"Technology",     peers:["MSFT","GOOGL","META"],price:213,vol:0.22,mu:0.14},
  NVDA:{name:"NVIDIA Corp.",     sector:"Semiconductors", peers:["AMD","INTC","AVGO"],  price:875,vol:0.55,mu:0.65},
  TSLA:{name:"Tesla Inc.",       sector:"EV / Energy",    peers:["F","GM","RIVN"],       price:245,vol:0.60,mu:0.20},
  MSFT:{name:"Microsoft Corp.",  sector:"Technology",     peers:["AAPL","GOOGL","ORCL"],price:415,vol:0.20,mu:0.18},
  AMZN:{name:"Amazon.com",       sector:"Cloud / E-Comm", peers:["MSFT","GOOGL","META"],price:198,vol:0.30,mu:0.22},
  META:{name:"Meta Platforms",   sector:"Social Media",   peers:["SNAP","GOOGL","PINS"],price:590,vol:0.35,mu:0.38},
  SPY: {name:"S&P 500 ETF",      sector:"Index",          peers:["QQQ","IWM","DIA"],    price:521,vol:0.15,mu:0.12},
  "BTC-USD":{name:"Bitcoin USD", sector:"Crypto",         peers:["ETH-USD","SOL-USD","BNB-USD"],price:85000,vol:0.70,mu:0.40},
};

// ═══════════════════════════════════════════════════════════════
//  CLAUDE HAIKU API
// ═══════════════════════════════════════════════════════════════
async function callHaiku(userPrompt,maxTokens=1200){
  const res=await fetch("https://api.anthropic.com/v1/messages",{
    method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({model:"claude-haiku-4-5-20251001",max_tokens:maxTokens,
      system:"You are a financial data API. Always respond with ONLY a valid JSON object. No prose, no markdown, no code fences. Raw JSON only.",
      tools:[{type:"web_search_20250305",name:"web_search"}],
      messages:[{role:"user",content:userPrompt}]}),
  });
  const data=await res.json();
  if(data.error) throw new Error(data.error.message||"API error");
  const allText=(data.content||[]).filter(b=>b.type==="text").map(b=>b.text).join("\n");
  if(!allText.trim()) throw new Error("Empty API response");
  const cleaned=allText.replace(/```json|```/g,"").trim();
  const s=cleaned.indexOf("{"),e=cleaned.lastIndexOf("}");
  if(s===-1||e<=s) throw new Error(`No JSON. Got: ${cleaned.slice(0,80)}`);
  return JSON.parse(cleaned.slice(s,e+1));
}

async function fetchStockData(ticker){
  const fb=STOCK_META[ticker]||{price:100,vol:0.30,mu:0.10,name:ticker,sector:"Equity",peers:["SPY"]};
  let rd={};
  try{
    rd=await callHaiku(`Search for the latest stock price and stats for ${ticker}. Fill in this JSON with real current values:
{"price":0,"high52w":0,"low52w":0,"ytdReturn":0,"impliedVol":0,"peRatio":0,"analystRating":"BUY","priceTarget":0,"marketCap":"0B"}
ytdReturn and impliedVol as decimals like 0.15 not 15. analystRating must be: STRONG BUY, BUY, HOLD, SELL, or STRONG SELL.`,1200);
  }catch(e){console.warn("Haiku fallback for",ticker,":",e.message);}
  const price=+(rd.price)||fb.price, vol=+(rd.impliedVol)||fb.vol, mu=+(rd.ytdReturn)||fb.mu;
  const seed=ticker.split("").reduce((a,c)=>a+c.charCodeAt(0),0);
  const {prices,dates}=simulateHistory(price,mu,vol,90,seed);
  return {ticker,prices,dates,currentPrice:price,name:fb.name,sector:fb.sector,peers:fb.peers,
    high52w:rd.high52w||null,low52w:rd.low52w||null,ytdReturn:rd.ytdReturn||null,
    peRatio:rd.peRatio||null,analystRating:rd.analystRating||null,
    priceTarget:rd.priceTarget||null,marketCap:rd.marketCap||null,impliedVol:vol};
}
async function fetchSentiment(ticker){
  return callHaiku(`Search for 5 recent news headlines about ${ticker} stock. Return this JSON with real headlines:
{"headlines":[{"title":"Actual headline","source":"Reuters","date":"Mar 20","sentiment":0.6,"summary":"One sentence.","tag":"EARNINGS"}],"overallSentiment":0.4,"sentimentLabel":"BULLISH","keyThemes":["theme1","theme2"]}
sentiment: -1 to +1. tag: EARNINGS, UPGRADE, DOWNGRADE, MACRO, PRODUCT, LEGAL, or GUIDANCE.`,1200);
}
async function fetchPeers(ticker){
  const peers=(STOCK_META[ticker]?.peers||["SPY","QQQ","IWM"]).slice(0,3);
  return callHaiku(`Search for current stock data for: ${[ticker,...peers].join(", ")}. Return this JSON:
{"peers":[{"ticker":"AAPL","name":"Apple","price":213,"ytdReturn":0.14,"peRatio":28,"analystRating":"BUY","annVol":0.22}]}
Include all ${peers.length+1} tickers. ytdReturn and annVol as decimals.`,1200);
}

// ═══════════════════════════════════════════════════════════════
//  UI HELPERS
// ═══════════════════════════════════════════════════════════════
const fmt=(v,d=2)=>typeof v==="number"?v.toFixed(d):"—";
const fmtPct=v=>typeof v==="number"?`${(v*100).toFixed(2)}%`:"—";
const fmtP=v=>typeof v==="number"?`$${v.toFixed(2)}`:"—";
const colV=(v,lo=0)=>v>lo?"#00e8a2":v<lo?"#ff4060":"#8899aa";
const RC={"STRONG BUY":"#00e8a2","BUY":"#4a9aff","HOLD":"#f0c040","SELL":"#ff8040","STRONG SELL":"#ff4060"};
const TT=({active,payload,label})=>{
  if(!active||!payload?.length)return null;
  return <div style={{background:"#08111d",border:"1px solid #162030",borderRadius:6,padding:"8px 12px",fontSize:11,fontFamily:"monospace"}}>
    <div style={{color:"#c8d8e8",marginBottom:4,fontWeight:600}}>{label}</div>
    {payload.map((p,i)=>p.value!=null&&<div key={i} style={{color:p.color||"#8899aa"}}>{p.name}: <span style={{color:"#e8edf3"}}>{typeof p.value==="number"?p.value.toFixed(3):p.value}</span></div>)}
  </div>;
};
const Chip=({label,value,color,sub})=>(
  <div style={{background:"#080f1a",border:`1px solid ${color||"#1a2535"}22`,borderRadius:8,padding:"12px 14px"}}>
    <div style={{fontSize:9,color:"#3a5068",letterSpacing:"0.12em",textTransform:"uppercase",fontFamily:"monospace",marginBottom:5}}>{label}</div>
    <div style={{fontSize:17,fontWeight:700,fontFamily:"monospace",color:color||"#d8e8f3",lineHeight:1}}>{value}</div>
    {sub&&<div style={{fontSize:9,color:"#2a4060",marginTop:4}}>{sub}</div>}
  </div>
);

// ═══════════════════════════════════════════════════════════════
//  MAIN APP
// ═══════════════════════════════════════════════════════════════
const TICKERS=["AAPL","NVDA","TSLA","MSFT","AMZN","META","SPY","BTC-USD"];
const TABS=["price","forecast","signals","risk","options","oscillators","backtest","sentiment","peers","portfolio"];

export default function QuantTerminal(){
  const [ticker,setTicker]=useState("NVDA");
  const [status,setStatus]=useState({phase:"idle",msg:""});
  const [analysis,setAnalysis]=useState(null);
  const [extras,setExtras]=useState({loading:false,sentiment:null,peers:null});
  const [tab,setTab]=useState("price");
  const [mcPaths,setMcPaths]=useState(300);
  const [optExp,setOptExp]=useState(30);
  const [optType,setOptType]=useState("call");
  const [portfolio,setPortfolio]=useState({cash:100000,positions:{},history:[]});
  const [tradeQty,setTradeQty]=useState("10");
  const [tradeMsg,setTradeMsg]=useState(null);

  const executeTrade=useCallback((type)=>{
    if(!analysis){setTradeMsg({type:"error",msg:"Load a stock first"});return;}
    const qty=parseInt(tradeQty)||0;
    if(qty<=0){setTradeMsg({type:"error",msg:"Enter a valid quantity"});return;}
    const price=analysis.S0, cost=price*qty;
    setPortfolio(prev=>{
      const pos={...prev.positions}; let cash=prev.cash;
      if(type==="buy"){
        if(cost>cash){setTradeMsg({type:"error",msg:`Need ${fmtP(cost)}, have ${fmtP(cash)}`});return prev;}
        cash-=cost;
        const ex=pos[ticker]||{shares:0,avgCost:0,totalCost:0};
        const ns=ex.shares+qty;
        pos[ticker]={shares:ns,avgCost:(ex.totalCost+cost)/ns,totalCost:ex.totalCost+cost};
        setTradeMsg({type:"success",msg:`Bought ${qty} ${ticker} @ ${fmtP(price)}`});
      } else {
        const hold=pos[ticker]?.shares||0;
        if(qty>hold){setTradeMsg({type:"error",msg:`Hold ${hold} shares, tried to sell ${qty}`});return prev;}
        cash+=price*qty;
        const ns=hold-qty;
        if(ns===0)delete pos[ticker];
        else pos[ticker]={...pos[ticker],shares:ns,totalCost:pos[ticker].totalCost-(pos[ticker].avgCost*qty)};
        setTradeMsg({type:"success",msg:`Sold ${qty} ${ticker} @ ${fmtP(price)}`});
      }
      setTimeout(()=>setTradeMsg(null),3000);
      return {cash,positions:pos,history:[...prev.history,{type,ticker,qty,price,time:new Date().toLocaleTimeString()}]};
    });
  },[analysis,ticker,tradeQty]);

  const runAnalysis=useCallback(async(t,paths)=>{
    setStatus({phase:"loading",msg:"Fetching live price data…"});
    setAnalysis(null); setExtras({loading:true,sentiment:null,peers:null});
    try{
      const [raw,sentRes,peerRes]=await Promise.all([
        fetchStockData(t),
        fetchSentiment(t).catch(e=>({error:e.message})),
        fetchPeers(t).catch(e=>({error:e.message})),
      ]);
      setExtras({loading:false,sentiment:sentRes,peers:peerRes});
      setStatus({phase:"loading",msg:"Running quant models…"});
      await new Promise(r=>setTimeout(r,10));
      const prices=raw.prices.map(Number).filter(v=>isFinite(v)&&v>0);
      const dates=(raw.dates||[]).slice(0,prices.length);
      if(prices.length<10) throw new Error("Not enough price data returned");
      const rets=logRet(prices);
      const garch=fitGARCH(rets), hmm=fitHMM(rets);
      const rsiVals=computeRSI(prices);
      const {macdLine,sigLine,hist:macdHist}=computeMACD(prices);
      const bbands=bollingerBands(prices), atrVals=atrFn(prices);
      const S0=prices[prices.length-1];
      setStatus({phase:"loading",msg:`Running ${paths} Monte Carlo paths…`});
      await new Promise(r=>setTimeout(r,10));
      const mc=garchMC(S0,rets,garch,hmm,paths);
      const forecastBands=Array.from({length:31},(_,i)=>{
        const vals=mc.map(p=>p[i]).sort((a,b)=>a-b),n=vals.length;
        return {day:i,p2:vals[Math.floor(n*.02)],p10:vals[Math.floor(n*.10)],p25:vals[Math.floor(n*.25)],p50:vals[Math.floor(n*.50)],p75:vals[Math.floor(n*.75)],p90:vals[Math.floor(n*.90)],p98:vals[Math.floor(n*.98)]};
      });
      const risk=riskMetrics(rets,prices);
      const signal=calibratedSignal(rets,prices,hmm,garch,rsiVals,bbands);
      const annSig=signal.garchVol*Math.sqrt(252)||0.3;
      const optionsChain=buildChain(S0,annSig);
      const backtest=runBacktest(prices,rets);
      setAnalysis({prices,dates,rets,rsiVals,macdLine,sigLine,macdHist,bbands,atrVals,forecastBands,garch,hmm,risk,signal,S0,meta:raw,paths,optionsChain,backtest});
      setStatus({phase:"done",msg:""});
    }catch(e){
      setStatus({phase:"error",msg:e.message});
      setExtras(x=>({...x,loading:false}));
    }
  },[]);

  useEffect(()=>{runAnalysis(ticker,mcPaths);},[ticker]);

  const priceData=analysis?analysis.prices.map((p,i)=>({date:analysis.dates[i]||`D${i}`,price:+p.toFixed(2),upper:analysis.bbands[i-19]?.upper!=null?+analysis.bbands[i-19].upper.toFixed(2):null,lower:analysis.bbands[i-19]?.lower!=null?+analysis.bbands[i-19].lower.toFixed(2):null,mid:analysis.bbands[i-19]?.mid!=null?+analysis.bbands[i-19].mid.toFixed(2):null})):[];
  const rsiData=analysis?analysis.rsiVals.map((r,i)=>({date:analysis.dates[i+15]||`D${i}`,rsi:+r.toFixed(1)})):[];
  const macdData=analysis?analysis.macdHist.map((h,i)=>({date:analysis.dates[i+34]||`D${i}`,hist:+h.toFixed(4),macd:analysis.macdLine[i+8]!=null?+analysis.macdLine[i+8].toFixed(4):null,signal:analysis.sigLine[i]!=null?+analysis.sigLine[i].toFixed(4):null})):[];
  const fcData=analysis?analysis.forecastBands.map((b,i)=>({day:`+${i}d`,p2:+b.p2?.toFixed(2),p10:+b.p10?.toFixed(2),p25:+b.p25?.toFixed(2),p50:+b.p50?.toFixed(2),p75:+b.p75?.toFixed(2),p90:+b.p90?.toFixed(2),p98:+b.p98?.toFixed(2)})):[];
  const volData=analysis?analysis.garch.condVar.map((v,i)=>({date:analysis.dates[i]||`D${i}`,vol:+(Math.sqrt(v)*Math.sqrt(252)*100).toFixed(2),lr:+(Math.sqrt(analysis.garch.longRunVar)*Math.sqrt(252)*100).toFixed(2)})):[];

  const sig=analysis?.signal;
  const sigStr=sig?.composite??0;
  const sigLabel=sigStr>0.4?"STRONG BUY":sigStr>0.1?"BUY":sigStr<-0.4?"STRONG SELL":sigStr<-0.1?"SELL":"NEUTRAL";
  const sigColor=sigStr>0.1?"#00e8a2":sigStr<-0.1?"#ff4060":"#f0c040";
  const regLabel=analysis?.hmm?.currentState===0?"BULL":"BEAR";
  const regColor=regLabel==="BULL"?"#00e8a2":"#ff4060";

  const totalPosVal=Object.entries(portfolio.positions).reduce((s,[sym,pos])=>{
    const p=sym===ticker?analysis?.S0:(STOCK_META[sym]?.price||0);
    return s+(p*pos.shares);
  },0);
  const totalVal=portfolio.cash+totalPosVal, totalPnL=totalVal-100000;

  return (
    <div style={{minHeight:"100vh",background:"#04090f",color:"#d0dde8",fontFamily:"'IBM Plex Sans',sans-serif"}}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap" rel="stylesheet"/>
      <style>{`*{box-sizing:border-box;margin:0;padding:0}::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-thumb{background:#1a2535;border-radius:2px}@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.35}}@keyframes fade{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}.fade{animation:fade 0.35s ease forwards}button:hover{opacity:0.85}`}</style>

      <div style={{borderBottom:"1px solid #0c1824",background:"#040c16",padding:"0 20px",display:"flex",alignItems:"center",height:48,gap:14}}>
        <div style={{fontFamily:"monospace",fontSize:11,color:"#0a6aff",letterSpacing:"0.18em",fontWeight:700}}>◈ ALPHA ENGINE</div>
        <div style={{width:1,height:16,background:"#0c1824"}}/>
        <div style={{fontSize:9,color:"#1e3a55",fontFamily:"monospace"}}>GARCH · HMM · MONTE CARLO · BLACK-SCHOLES · PAPER TRADING</div>
        <div style={{flex:1}}/>
        <span style={{fontSize:9,color:"#0a2535",fontFamily:"monospace"}}>PORTFOLIO:</span>
        <span style={{fontSize:12,fontFamily:"monospace",color:totalPnL>=0?"#00e8a2":"#ff4060",fontWeight:700}}>${totalVal.toFixed(0)}</span>
        <span style={{fontSize:9,fontFamily:"monospace",color:totalPnL>=0?"#00e8a2":"#ff4060"}}>{totalPnL>=0?"+":""}{fmtPct(totalPnL/100000)}</span>
      </div>

      <div style={{padding:"14px 20px 28px",maxWidth:1160,margin:"0 auto"}}>
        <div style={{display:"flex",gap:5,marginBottom:14,flexWrap:"wrap",alignItems:"center"}}>
          {TICKERS.map(t=><button key={t} onClick={()=>setTicker(t)} style={{padding:"4px 12px",borderRadius:4,border:`1px solid ${ticker===t?"#0a5adf":"#0c1824"}`,background:ticker===t?"#051830":"#060e1a",color:ticker===t?"#4a9aff":"#2a4560",fontFamily:"monospace",fontSize:11,fontWeight:600,cursor:"pointer"}}>{t}</button>)}
          <div style={{flex:1}}/>
          <span style={{fontSize:9,color:"#1a3050",fontFamily:"monospace"}}>PATHS:</span>
          {[100,300,500].map(n=><button key={n} onClick={()=>setMcPaths(n)} style={{padding:"2px 8px",fontSize:9,fontFamily:"monospace",border:`1px solid ${mcPaths===n?"#0a5adf":"#0c1824"}`,background:mcPaths===n?"#051830":"transparent",color:mcPaths===n?"#4a9aff":"#1a3050",borderRadius:3,cursor:"pointer"}}>{n}</button>)}
          <button onClick={()=>runAnalysis(ticker,mcPaths)} style={{padding:"4px 12px",borderRadius:4,border:"1px solid #0a2540",background:"#060e1a",color:"#0a6aff",fontFamily:"monospace",fontSize:10,cursor:"pointer"}}>{status.phase==="loading"?"⟳ RUNNING":"↺ RERUN"}</button>
        </div>

        {status.phase==="loading"&&<div style={{padding:"50px 0",textAlign:"center"}}>
          <div style={{fontSize:11,fontFamily:"monospace",color:"#0a5adf",letterSpacing:"0.12em",animation:"pulse 1.4s ease infinite"}}>{status.msg}</div>
          <div style={{marginTop:10,fontSize:9,color:"#0a2535",fontFamily:"monospace"}}>Web search · GARCH calibration · HMM regime detection · {mcPaths}-path Monte Carlo</div>
        </div>}

        {status.phase==="error"&&<div style={{padding:"30px 0",textAlign:"center"}}>
          <div style={{color:"#ff4060",fontFamily:"monospace",fontSize:12}}>✕ {status.msg}</div>
          <button onClick={()=>runAnalysis(ticker,mcPaths)} style={{marginTop:10,background:"transparent",border:"1px solid #4a2030",color:"#ff4060",fontFamily:"monospace",padding:"5px 14px",borderRadius:4,cursor:"pointer"}}>Try Again</button>
        </div>}

        {status.phase==="done"&&analysis&&<div className="fade">
          {/* HEADER */}
          <div style={{display:"flex",gap:10,marginBottom:14,alignItems:"flex-start",flexWrap:"wrap"}}>
            <div style={{flex:1,minWidth:150}}>
              <div style={{fontSize:26,fontWeight:700,letterSpacing:"-0.03em",fontFamily:"monospace"}}>{ticker}</div>
              <div style={{fontSize:11,color:"#2a5070",marginTop:1}}>{analysis.meta.name} · {analysis.meta.sector}</div>
              <div style={{fontSize:22,fontWeight:600,color:"#d0dde8",marginTop:4,fontFamily:"monospace"}}>{fmtP(analysis.S0)}</div>
            </div>
            <div style={{padding:"9px 18px",borderRadius:8,border:`1px solid ${sigColor}30`,background:`${sigColor}08`,textAlign:"center"}}>
              <div style={{fontSize:8,color:"#2a4060",letterSpacing:"0.14em",fontFamily:"monospace",marginBottom:2}}>SIGNAL</div>
              <div style={{fontSize:17,fontWeight:800,color:sigColor,fontFamily:"monospace"}}>{sigLabel}</div>
              <div style={{fontSize:9,color:sigColor+"80",fontFamily:"monospace"}}>{(sigStr*100).toFixed(0)}/100</div>
            </div>
            <div style={{padding:"9px 18px",borderRadius:8,border:`1px solid ${regColor}25`,background:`${regColor}07`,textAlign:"center"}}>
              <div style={{fontSize:8,color:"#2a4060",letterSpacing:"0.14em",fontFamily:"monospace",marginBottom:2}}>REGIME</div>
              <div style={{fontSize:17,fontWeight:800,color:regColor,fontFamily:"monospace"}}>{regLabel}</div>
              <div style={{fontSize:9,color:regColor+"80",fontFamily:"monospace"}}>P(bull)={fmtPct(sig?.bullProb)}</div>
            </div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(2,1fr)",gap:5}}>
              <Chip label="Sharpe" value={fmt(analysis.risk.sharpe)} color={colV(analysis.risk.sharpe,1)} sub="annualized"/>
              <Chip label="GARCH Vol" value={fmtPct(sig?.garchVol*Math.sqrt(252))} color="#4a9aff" sub="current"/>
              <Chip label="Max DD" value={fmtPct(-analysis.risk.mdd)} color="#ff4060"/>
              {analysis.meta.analystRating
                ?<Chip label="Analyst" value={analysis.meta.analystRating} color={RC[analysis.meta.analystRating]||"#8899aa"} sub={analysis.meta.priceTarget?`target ${fmtP(analysis.meta.priceTarget)}`:""}/>
                :<Chip label="Sortino" value={fmt(analysis.risk.sortino)} color={colV(analysis.risk.sortino,1)}/>}
            </div>
          </div>

          {/* TABS */}
          <div style={{display:"flex",borderBottom:"1px solid #0c1824",marginBottom:12,overflowX:"auto"}}>
            {TABS.map(t=><button key={t} onClick={()=>setTab(t)} style={{padding:"6px 13px",border:"none",borderBottom:`2px solid ${tab===t?"#0a6aff":"transparent"}`,background:"transparent",color:tab===t?"#4a9aff":"#1e3a55",fontSize:9,fontFamily:"monospace",letterSpacing:"0.08em",textTransform:"uppercase",cursor:"pointer",whiteSpace:"nowrap",marginBottom:-1}}>{t}</button>)}
          </div>

          {/* PRICE */}
          {tab==="price"&&<div>
            <div style={{fontSize:9,color:"#1a3550",fontFamily:"monospace",marginBottom:6}}>PRICE + BOLLINGER BANDS (20,2)</div>
            <div style={{height:280,background:"#060e1a",borderRadius:8,border:"1px solid #0c1824",padding:"10px 4px"}}>
              <ResponsiveContainer><LineChart data={priceData}>
                <XAxis dataKey="date" tick={{fill:"#1a3050",fontSize:7,fontFamily:"monospace"}} tickLine={false} axisLine={false} interval={17}/>
                <YAxis tick={{fill:"#1a3050",fontSize:7,fontFamily:"monospace"}} tickLine={false} axisLine={false} tickFormatter={v=>`$${v}`} domain={["auto","auto"]}/>
                <Tooltip content={<TT/>}/>
                <Line type="monotone" dataKey="upper" stroke="#1a3550" strokeWidth={1} dot={false} name="BB Upper" strokeDasharray="2 3"/>
                <Line type="monotone" dataKey="lower" stroke="#1a3550" strokeWidth={1} dot={false} name="BB Lower" strokeDasharray="2 3"/>
                <Line type="monotone" dataKey="mid" stroke="#0a2535" strokeWidth={1} dot={false} name="BB Mid"/>
                <Line type="monotone" dataKey="price" stroke="#0a6aff" strokeWidth={2} dot={false} name="Price"/>
              </LineChart></ResponsiveContainer>
            </div>
            <div style={{marginTop:6,display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:5}}>
              {[{label:"52W High",value:analysis.meta.high52w?fmtP(analysis.meta.high52w):fmtP(Math.max(...analysis.prices))},
                {label:"52W Low",value:analysis.meta.low52w?fmtP(analysis.meta.low52w):fmtP(Math.min(...analysis.prices))},
                {label:"ATR (14)",value:fmtP(analysis.atrVals[analysis.atrVals.length-1]||0)},
                {label:"BB %B",value:fmt(analysis.signal.currentBB?.pctB)},
              ].map(m=><Chip key={m.label} {...m}/>)}
            </div>
          </div>}

          {/* FORECAST */}
          {tab==="forecast"&&<div>
            <div style={{fontSize:9,color:"#1a3550",fontFamily:"monospace",marginBottom:6}}>GARCH MONTE CARLO · {analysis.paths} PATHS · 30-DAY · REGIME-CONDITIONAL</div>
            <div style={{height:280,background:"#060e1a",borderRadius:8,border:"1px solid #0c1824",padding:"10px 4px"}}>
              <ResponsiveContainer><LineChart data={fcData}>
                <XAxis dataKey="day" tick={{fill:"#1a3050",fontSize:7,fontFamily:"monospace"}} tickLine={false} axisLine={false}/>
                <YAxis tick={{fill:"#1a3050",fontSize:7,fontFamily:"monospace"}} tickLine={false} axisLine={false} tickFormatter={v=>`$${v?.toFixed(0)}`} domain={["auto","auto"]}/>
                <Tooltip content={<TT/>}/>
                <Line type="monotone" dataKey="p90" stroke="#0a3020" strokeWidth={1.5} dot={false} name="P90"/>
                <Line type="monotone" dataKey="p75" stroke="#0d5a30" strokeWidth={2} dot={false} name="P75"/>
                <Line type="monotone" dataKey="p50" stroke="#00e8a2" strokeWidth={2.5} dot={false} name="Median"/>
                <Line type="monotone" dataKey="p25" stroke="#6a1a1a" strokeWidth={2} dot={false} name="P25"/>
                <Line type="monotone" dataKey="p10" stroke="#3a0e0e" strokeWidth={1.5} dot={false} name="P10"/>
                <ReferenceLine y={analysis.S0} stroke="#1a2535" strokeDasharray="3 3"/>
              </LineChart></ResponsiveContainer>
            </div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:5,marginTop:6}}>
              {[{l:"Bull (P90)",v:fcData[30]?.p90},{l:"Opt (P75)",v:fcData[30]?.p75},{l:"Base (P50)",v:fcData[30]?.p50},{l:"Pess (P25)",v:fcData[30]?.p25},{l:"Bear (P10)",v:fcData[30]?.p10}].map(({l,v})=>(
                <div key={l} style={{background:"#060e1a",border:"1px solid #0c1824",borderRadius:6,padding:"9px 11px"}}>
                  <div style={{fontSize:8,color:"#1a3050",fontFamily:"monospace",marginBottom:2}}>{l}</div>
                  <div style={{fontSize:13,fontWeight:700,fontFamily:"monospace",color:v>=analysis.S0?"#00e8a2":"#ff4060"}}>{fmtP(v)}</div>
                  <div style={{fontSize:8,color:"#1a3050",fontFamily:"monospace"}}>{v?`${((v-analysis.S0)/analysis.S0*100).toFixed(1)}%`:""}</div>
                </div>
              ))}
            </div>
          </div>}

          {/* SIGNALS */}
          {tab==="signals"&&sig&&<div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
            <div style={{background:"#060e1a",border:"1px solid #0c1824",borderRadius:8,padding:16}}>
              <div style={{fontSize:9,color:"#1a3550",fontFamily:"monospace",marginBottom:12}}>MULTI-FACTOR DECOMPOSITION</div>
              {[{label:"TREND (UMD)",score:sig.trendScore,w:"35%",desc:"12M minus 1M momentum"},
                {label:"REVERSION (RSI+BB)",score:sig.reversionScore,w:"20%",desc:"RSI + Bollinger %B"},
                {label:"REGIME (HMM)",score:sig.regimeScore,w:"25%",desc:`P(bull)=${fmtPct(sig.bullProb)}`},
                {label:"IR MOMENTUM",score:sig.irScore,w:"20%",desc:"5D return / 20D vol"},
              ].map(({label,score,w,desc})=>{
                const pct=clamp(score,-1,1),col=pct>0.1?"#00e8a2":pct<-0.1?"#ff4060":"#f0c040";
                return <div key={label} style={{marginBottom:10}}>
                  <div style={{display:"flex",justifyContent:"space-between",marginBottom:2}}>
                    <span style={{fontSize:9,color:"#5a7090",fontFamily:"monospace"}}>{label} <span style={{color:"#2a4060"}}>w={w}</span></span>
                    <span style={{fontSize:9,fontFamily:"monospace",color:col}}>{pct>0?"+":""}{(pct*100).toFixed(1)}</span>
                  </div>
                  <div style={{height:4,background:"#0a1520",borderRadius:2,position:"relative"}}>
                    <div style={{position:"absolute",left:"50%",top:0,width:1,height:"100%",background:"#1a2535"}}/>
                    <div style={{position:"absolute",height:"100%",borderRadius:2,background:col,left:pct>=0?"50%":`${(0.5+pct/2)*100}%`,width:`${Math.abs(pct)/2*100}%`}}/>
                  </div>
                  <div style={{fontSize:8,color:"#1e3045",marginTop:1,fontFamily:"monospace"}}>{desc}</div>
                </div>;
              })}
              <div style={{height:1,background:"#0c1824",margin:"10px 0"}}/>
              <div style={{display:"flex",alignItems:"center",gap:10}}>
                <div style={{width:56,height:56,borderRadius:"50%",border:`2px solid ${sigColor}`,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",background:`${sigColor}08`}}>
                  <div style={{fontSize:13,fontWeight:700,color:sigColor,fontFamily:"monospace"}}>{(sigStr*100).toFixed(0)}</div>
                  <div style={{fontSize:6,color:sigColor+"70",fontFamily:"monospace"}}>SCORE</div>
                </div>
                <div>
                  <div style={{fontSize:15,fontWeight:800,color:sigColor,fontFamily:"monospace"}}>{sigLabel}</div>
                  <div style={{fontSize:8,color:"#1a3050",fontFamily:"monospace"}}>Vol penalty {fmt(sig.volPenalty)}× · Half-Kelly {fmt(sig.kelly*100)}%</div>
                </div>
              </div>
            </div>
            <div style={{display:"flex",flexDirection:"column",gap:10}}>
              <div style={{background:"#060e1a",border:"1px solid #0c1824",borderRadius:8,padding:14,flex:1}}>
                <div style={{fontSize:9,color:"#1a3550",fontFamily:"monospace",marginBottom:8}}>GARCH CONDITIONAL VOL</div>
                <div style={{height:120}}>
                  <ResponsiveContainer><LineChart data={volData}>
                    <XAxis dataKey="date" tick={{fill:"#1a3050",fontSize:6,fontFamily:"monospace"}} tickLine={false} axisLine={false} interval={19}/>
                    <YAxis tick={{fill:"#1a3050",fontSize:6,fontFamily:"monospace"}} tickLine={false} axisLine={false} tickFormatter={v=>`${v}%`}/>
                    <Tooltip content={<TT/>}/>
                    <Line type="monotone" dataKey="vol" stroke="#4a9aff" strokeWidth={1.5} dot={false} name="Cond. Vol"/>
                    <Line type="monotone" dataKey="lr" stroke="#1a3550" strokeWidth={1} dot={false} name="Long-run" strokeDasharray="3 3"/>
                  </LineChart></ResponsiveContainer>
                </div>
              </div>
              <div style={{background:"#060e1a",border:"1px solid #0c1824",borderRadius:8,padding:14}}>
                <div style={{fontSize:9,color:"#1a3550",fontFamily:"monospace",marginBottom:6}}>KEY INPUTS</div>
                {[["RSI(14)",fmt(sig.currentRSI,1),sig.currentRSI<30?"#00e8a2":sig.currentRSI>70?"#ff4060":"#4a9aff"],
                  ["5D Mom",fmtPct(sig.mf.mom5),colV(sig.mf.mom5)],
                  ["BB %B",fmt(sig.currentBB?.pctB),sig.currentBB?.pctB>0.8?"#ff4060":sig.currentBB?.pctB<0.2?"#00e8a2":"#4a9aff"],
                  ["GARCH Vol",fmtPct(sig.garchVol),"#4a9aff"],["α+β",fmt(sig.persistence),sig.persistence>0.97?"#ff4060":"#f0c040"],
                ].map(([l,v,c])=>(
                  <div key={l} style={{display:"flex",justifyContent:"space-between",padding:"3px 0",borderBottom:"1px solid #0a1420",fontSize:9,fontFamily:"monospace"}}>
                    <span style={{color:"#2a4060"}}>{l}</span><span style={{color:c}}>{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>}

          {/* RISK */}
          {tab==="risk"&&<div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:5,marginBottom:10}}>
              {[{label:"Sharpe",value:fmt(analysis.risk.sharpe),color:colV(analysis.risk.sharpe,1),sub:"ann."},
                {label:"Sortino",value:fmt(analysis.risk.sortino),color:colV(analysis.risk.sortino,1),sub:"downside"},
                {label:"Calmar",value:fmt(analysis.risk.calmar),color:colV(analysis.risk.calmar,0.5),sub:"ret/maxDD"},
                {label:"Max DD",value:fmtPct(-analysis.risk.mdd),color:"#ff4060"},
                {label:"Ann. Return",value:fmtPct(analysis.risk.annRet),color:colV(analysis.risk.annRet)},
                {label:"Ann. Vol",value:fmtPct(analysis.risk.annVol),color:"#4a9aff"},
                {label:"VaR 95%",value:fmtPct(analysis.risk.var95),color:"#ff4060",sub:"1-day"},
                {label:"CVaR 95%",value:fmtPct(analysis.risk.cvar95),color:"#ff4060",sub:"exp. shortfall"},
              ].map(m=><Chip key={m.label} {...m}/>)}
            </div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8}}>
              <div style={{background:"#060e1a",border:"1px solid #0c1824",borderRadius:6,padding:14}}>
                <div style={{fontSize:9,color:"#1a3550",fontFamily:"monospace",marginBottom:6}}>DISTRIBUTION</div>
                {[["Skewness",fmt(analysis.risk.skew,3),"Neg = fat left tail"],["Excess Kurt",fmt(analysis.risk.kurt,3),">0 = fat tails"]].map(([l,v,d])=>(
                  <div key={l} style={{padding:"4px 0",borderBottom:"1px solid #0a1420"}}>
                    <div style={{display:"flex",justifyContent:"space-between",fontSize:10,fontFamily:"monospace"}}><span style={{color:"#2a4060"}}>{l}</span><span style={{color:"#4a9aff"}}>{v}</span></div>
                    <div style={{fontSize:8,color:"#0a2030",fontFamily:"monospace"}}>{d}</div>
                  </div>
                ))}
              </div>
              <div style={{background:"#060e1a",border:"1px solid #0c1824",borderRadius:6,padding:14}}>
                <div style={{fontSize:9,color:"#1a3550",fontFamily:"monospace",marginBottom:6}}>GARCH PARAMS</div>
                {[["ω",fmt(analysis.garch.omega,8)],["α",fmt(analysis.garch.alpha)],["β",fmt(analysis.garch.beta)],["α+β",fmt(analysis.garch.persistence)],["Long-run var",fmt(analysis.garch.longRunVar,8)]].map(([l,v])=>(
                  <div key={l} style={{display:"flex",justifyContent:"space-between",padding:"3px 0",borderBottom:"1px solid #0a1420",fontSize:9,fontFamily:"monospace"}}><span style={{color:"#2a4060"}}>{l}</span><span style={{color:"#4a9aff"}}>{v}</span></div>
                ))}
              </div>
            </div>
          </div>}

          {/* OPTIONS */}
          {tab==="options"&&analysis.optionsChain&&(()=>{
            const chain=analysis.optionsChain.find(c=>c.expiry===optExp)||analysis.optionsChain[2];
            const atm=chain.strikes.reduce((b,r)=>Math.abs(r.K-analysis.S0)<Math.abs(b.K-analysis.S0)?r:b);
            const o=atm[optType];
            return <div>
              <div style={{display:"flex",gap:6,marginBottom:8,alignItems:"center",flexWrap:"wrap"}}>
                <div style={{fontSize:9,color:"#1a3550",fontFamily:"monospace",flex:1}}>BLACK-SCHOLES · IV={fmtPct(analysis.meta.impliedVol)} · rf=5.25%</div>
                {["call","put"].map(t=><button key={t} onClick={()=>setOptType(t)} style={{padding:"2px 10px",fontSize:9,fontFamily:"monospace",border:`1px solid ${optType===t?"#0a6aff":"#0c1824"}`,background:optType===t?"#051830":"transparent",color:optType===t?"#4a9aff":"#2a4560",borderRadius:4,cursor:"pointer",textTransform:"uppercase"}}>{t}</button>)}
                {[7,14,30,60,90].map(d=><button key={d} onClick={()=>setOptExp(d)} style={{padding:"2px 8px",fontSize:9,fontFamily:"monospace",border:`1px solid ${optExp===d?"#0a6aff":"#0c1824"}`,background:optExp===d?"#051830":"transparent",color:optExp===d?"#4a9aff":"#2a4560",borderRadius:4,cursor:"pointer"}}>{d}D</button>)}
              </div>
              <div style={{overflowX:"auto"}}>
                <table style={{width:"100%",borderCollapse:"collapse",fontFamily:"monospace",fontSize:9}}>
                  <thead><tr style={{borderBottom:"1px solid #0c1824"}}>
                    {["Strike","OTM%","Price","Δ","Γ","Θ","ν","ρ"].map(h=><th key={h} style={{padding:"5px 8px",color:"#1a3550",textAlign:"right",fontWeight:600}}>{h}</th>)}
                  </tr></thead>
                  <tbody>
                    {chain.strikes.map((row,i)=>{
                      const opt=row[optType],atm2=Math.abs(row.K-analysis.S0)<analysis.S0*0.03,itm=optType==="call"?row.K<analysis.S0:row.K>analysis.S0;
                      return <tr key={i} style={{background:atm2?"#0a2040":itm?"#060e1a":"transparent",borderBottom:"1px solid #080f1a"}}>
                        <td style={{padding:"4px 8px",color:atm2?"#4a9aff":"#8899aa",fontWeight:atm2?700:400}}>${row.K}</td>
                        <td style={{padding:"4px 8px",textAlign:"right",color:parseFloat(row.moneyness)>0?"#00e8a2":parseFloat(row.moneyness)<0?"#ff4060":"#f0c040"}}>{row.moneyness}%</td>
                        <td style={{padding:"4px 8px",textAlign:"right",color:"#d0dde8"}}>${opt.price.toFixed(2)}</td>
                        <td style={{padding:"4px 8px",textAlign:"right",color:opt.delta>0?"#00e8a2":"#ff4060"}}>{opt.delta.toFixed(3)}</td>
                        <td style={{padding:"4px 8px",textAlign:"right",color:"#4a9aff"}}>{opt.gamma.toFixed(4)}</td>
                        <td style={{padding:"4px 8px",textAlign:"right",color:"#ff4060"}}>{opt.theta.toFixed(3)}</td>
                        <td style={{padding:"4px 8px",textAlign:"right",color:"#f0c040"}}>{opt.vega.toFixed(3)}</td>
                        <td style={{padding:"4px 8px",textAlign:"right",color:"#8899aa"}}>{opt.rho.toFixed(3)}</td>
                      </tr>;
                    })}
                  </tbody>
                </table>
              </div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:5,marginTop:8}}>
                {[{label:"ATM Price",value:`$${o.price.toFixed(2)}`},{label:"Delta Δ",value:o.delta.toFixed(3),color:o.delta>0?"#00e8a2":"#ff4060",sub:"≈ prob ITM"},{label:"Gamma Γ",value:o.gamma.toFixed(4),color:"#4a9aff"},{label:"Theta Θ",value:`$${o.theta.toFixed(3)}/d`,color:"#ff4060"},{label:"Vega ν",value:`$${o.vega.toFixed(3)}`,color:"#f0c040"}].map(m=><Chip key={m.label} {...m}/>)}
              </div>
            </div>;
          })()}

          {/* OSCILLATORS */}
          {tab==="oscillators"&&<div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
            <div>
              <div style={{fontSize:9,color:"#1a3550",fontFamily:"monospace",marginBottom:5}}>RSI (14) — {fmt(sig?.currentRSI,1)}</div>
              <div style={{height:190,background:"#060e1a",borderRadius:8,border:"1px solid #0c1824",padding:"8px 4px"}}>
                <ResponsiveContainer><AreaChart data={rsiData}>
                  <defs><linearGradient id="rg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#0a6aff" stopOpacity={0.2}/><stop offset="95%" stopColor="#0a6aff" stopOpacity={0}/></linearGradient></defs>
                  <XAxis dataKey="date" tick={{fill:"#1a3050",fontSize:6,fontFamily:"monospace"}} tickLine={false} axisLine={false} interval={18}/>
                  <YAxis domain={[0,100]} tick={{fill:"#1a3050",fontSize:6,fontFamily:"monospace"}} tickLine={false} axisLine={false}/>
                  <Tooltip content={<TT/>}/>
                  <ReferenceLine y={70} stroke="#ff406044" strokeDasharray="2 3"/>
                  <ReferenceLine y={30} stroke="#00e8a244" strokeDasharray="2 3"/>
                  <Area type="monotone" dataKey="rsi" stroke="#0a6aff" strokeWidth={1.5} fill="url(#rg)" name="RSI" dot={false}/>
                </AreaChart></ResponsiveContainer>
              </div>
            </div>
            <div>
              <div style={{fontSize:9,color:"#1a3550",fontFamily:"monospace",marginBottom:5}}>MACD (12,26,9)</div>
              <div style={{height:190,background:"#060e1a",borderRadius:8,border:"1px solid #0c1824",padding:"8px 4px"}}>
                <ResponsiveContainer><ComposedChart data={macdData}>
                  <XAxis dataKey="date" tick={{fill:"#1a3050",fontSize:6,fontFamily:"monospace"}} tickLine={false} axisLine={false} interval={18}/>
                  <YAxis tick={{fill:"#1a3050",fontSize:6,fontFamily:"monospace"}} tickLine={false} axisLine={false}/>
                  <Tooltip content={<TT/>}/>
                  <ReferenceLine y={0} stroke="#1a2535"/>
                  <Bar dataKey="hist" name="Hist" shape={p=><rect x={p.x} y={p.y} width={p.width} height={Math.abs(p.height||0)} fill={p.value>=0?"#00e8a2":"#ff4060"} opacity={0.7}/>}/>
                  <Line type="monotone" dataKey="macd" stroke="#4a9aff" strokeWidth={1} dot={false} name="MACD"/>
                  <Line type="monotone" dataKey="signal" stroke="#f0c040" strokeWidth={1} dot={false} name="Signal"/>
                </ComposedChart></ResponsiveContainer>
              </div>
            </div>
          </div>}

          {/* BACKTEST */}
          {tab==="backtest"&&(()=>{
            const bt=analysis.backtest;
            return <div>
              <div style={{fontSize:9,color:"#1a3550",fontFamily:"monospace",marginBottom:6}}>EMA 10/30 GOLDEN CROSS · $10,000 CAPITAL · LONG-ONLY</div>
              <div style={{height:250,background:"#060e1a",borderRadius:8,border:"1px solid #0c1824",padding:"10px 4px"}}>
                <ResponsiveContainer><LineChart data={bt.equity}>
                  <XAxis dataKey="day" tick={{fill:"#1a3050",fontSize:7,fontFamily:"monospace"}} tickLine={false} axisLine={false} interval={Math.max(Math.floor(bt.equity.length/6),1)}/>
                  <YAxis tick={{fill:"#1a3050",fontSize:7,fontFamily:"monospace"}} tickLine={false} axisLine={false} tickFormatter={v=>`$${(v/1000).toFixed(1)}k`} domain={["auto","auto"]}/>
                  <Tooltip content={<TT/>}/>
                  <ReferenceLine y={10000} stroke="#1a2535" strokeDasharray="3 3"/>
                  <Line type="monotone" dataKey="benchmark" stroke="#2a4060" strokeWidth={1.5} dot={false} name="Buy & Hold"/>
                  <Line type="monotone" dataKey="value" stroke="#00e8a2" strokeWidth={2} dot={false} name="Strategy"/>
                </LineChart></ResponsiveContainer>
              </div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:5,marginTop:8}}>
                {[{label:"Return",value:fmtPct(bt.totalReturn),color:colV(bt.totalReturn)},{label:"Benchmark",value:fmtPct(bt.bmReturn),color:colV(bt.bmReturn),sub:"buy & hold"},{label:"Alpha",value:fmtPct(bt.alpha),color:colV(bt.alpha)},{label:"Sharpe",value:fmt(bt.sharpe),color:colV(bt.sharpe,1)},{label:"Max DD",value:fmtPct(-bt.maxDD),color:"#ff4060"},{label:"Ann. Return",value:fmtPct(bt.annRet),color:colV(bt.annRet)},{label:"Win Rate",value:fmtPct(bt.winRate),color:colV(bt.winRate,0.5),sub:`${bt.nTrades} trades`},{label:"Ann. Vol",value:fmtPct(bt.annVol),color:"#4a9aff"}].map(m=><Chip key={m.label} {...m}/>)}
              </div>
            </div>;
          })()}

          {/* SENTIMENT */}
          {tab==="sentiment"&&<div>
            {extras.loading&&<div style={{padding:"30px 0",textAlign:"center",fontSize:10,color:"#0a5adf",fontFamily:"monospace",animation:"pulse 1.4s ease infinite"}}>Fetching news sentiment…</div>}
            {extras.sentiment?.error&&<div style={{padding:"12px 0",color:"#ff4060",fontFamily:"monospace",fontSize:11}}>✕ {extras.sentiment.error}</div>}
            {extras.sentiment&&!extras.sentiment.error&&(()=>{
              const s=extras.sentiment,sc=typeof s.overallSentiment==="number"?s.overallSentiment:0;
              const scCol=sc>0.2?"#00e8a2":sc<-0.2?"#ff4060":"#f0c040";
              const TC={EARNINGS:"#4a9aff",UPGRADE:"#00e8a2",DOWNGRADE:"#ff4060",MACRO:"#f0c040",PRODUCT:"#a070ff",LEGAL:"#ff8040",GUIDANCE:"#4a9aff"};
              return <div>
                <div style={{display:"flex",gap:10,marginBottom:12,alignItems:"center"}}>
                  <div style={{padding:"8px 16px",borderRadius:8,border:`1px solid ${scCol}30`,background:`${scCol}08`,textAlign:"center"}}>
                    <div style={{fontSize:8,color:"#2a4060",fontFamily:"monospace",letterSpacing:"0.12em",marginBottom:2}}>SENTIMENT</div>
                    <div style={{fontSize:18,fontWeight:800,color:scCol,fontFamily:"monospace"}}>{s.sentimentLabel||"NEUTRAL"}</div>
                    <div style={{fontSize:9,color:scCol+"80",fontFamily:"monospace"}}>{sc>0?"+":""}{(sc*100).toFixed(0)}/100</div>
                  </div>
                  <div style={{flex:1}}>
                    <div style={{fontSize:9,color:"#1a3550",fontFamily:"monospace",marginBottom:5}}>KEY THEMES</div>
                    <div style={{display:"flex",gap:5,flexWrap:"wrap"}}>{(s.keyThemes||[]).map((t,i)=><span key={i} style={{padding:"2px 8px",borderRadius:3,background:"#0a1a2a",border:"1px solid #1a3050",fontSize:9,color:"#4a7090",fontFamily:"monospace"}}>{t}</span>)}</div>
                  </div>
                </div>
                {(s.headlines||[]).map((h,i)=>{
                  const c=h.sentiment>0.2?"#00e8a2":h.sentiment<-0.2?"#ff4060":"#f0c040",tc=TC[h.tag]||"#4a7090";
                  return <div key={i} style={{background:"#060e1a",border:"1px solid #0c1824",borderRadius:8,padding:"10px 12px",display:"flex",gap:10,marginBottom:6}}>
                    <div style={{width:32,height:32,borderRadius:"50%",border:`2px solid ${c}`,display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,background:`${c}08`}}>
                      <span style={{fontSize:9,fontWeight:700,color:c,fontFamily:"monospace"}}>{h.sentiment>0?"+":""}{(h.sentiment*10).toFixed(0)}</span>
                    </div>
                    <div style={{flex:1,minWidth:0}}>
                      <div style={{display:"flex",gap:5,marginBottom:3,flexWrap:"wrap"}}>
                        <span style={{fontSize:7,padding:"1px 5px",borderRadius:2,background:`${tc}15`,color:tc,fontFamily:"monospace",border:`1px solid ${tc}30`}}>{h.tag}</span>
                        <span style={{fontSize:7,color:"#1a3550",fontFamily:"monospace"}}>{h.source}</span>
                        <span style={{fontSize:7,color:"#0a2535",fontFamily:"monospace"}}>{h.date}</span>
                      </div>
                      <div style={{fontSize:10,color:"#8899aa",marginBottom:2,lineHeight:1.4}}>{h.title}</div>
                      <div style={{fontSize:8,color:"#2a4060",fontFamily:"monospace"}}>{h.summary}</div>
                    </div>
                  </div>;
                })}
              </div>;
            })()}
          </div>}

          {/* PEERS */}
          {tab==="peers"&&<div>
            {extras.loading&&<div style={{padding:"30px 0",textAlign:"center",fontSize:10,color:"#0a5adf",fontFamily:"monospace",animation:"pulse 1.4s ease infinite"}}>Fetching peer data…</div>}
            {extras.peers?.error&&<div style={{padding:"12px 0",color:"#ff4060",fontFamily:"monospace",fontSize:11}}>✕ {extras.peers.error}</div>}
            {extras.peers?.peers&&(()=>{
              const peers=extras.peers.peers;
              return <div>
                <div style={{overflowX:"auto",marginBottom:10}}>
                  <table style={{width:"100%",borderCollapse:"collapse",fontFamily:"monospace",fontSize:9}}>
                    <thead><tr style={{borderBottom:"1px solid #0c1824"}}>
                      {["Ticker","Name","Price","YTD","P/E","Rating","Ann. Vol"].map(h=><th key={h} style={{padding:"5px 8px",color:"#1a3550",textAlign:"right",fontWeight:600,fontSize:8}}>{h}</th>)}
                    </tr></thead>
                    <tbody>{peers.map((p,i)=>(
                      <tr key={i} style={{background:p.ticker===ticker?"#0a2040":"transparent",borderBottom:"1px solid #080f1a"}}>
                        <td style={{padding:"5px 8px",color:p.ticker===ticker?"#4a9aff":"#8899aa",fontWeight:p.ticker===ticker?700:400}}>{p.ticker}</td>
                        <td style={{padding:"5px 8px",color:"#5a7090",whiteSpace:"nowrap"}}>{p.name}</td>
                        <td style={{padding:"5px 8px",textAlign:"right",color:"#d0dde8"}}>{p.price?fmtP(p.price):"—"}</td>
                        <td style={{padding:"5px 8px",textAlign:"right",color:colV(p.ytdReturn)}}>{p.ytdReturn!=null?fmtPct(p.ytdReturn):"—"}</td>
                        <td style={{padding:"5px 8px",textAlign:"right",color:"#8899aa"}}>{p.peRatio?p.peRatio.toFixed(1):"—"}</td>
                        <td style={{padding:"5px 8px",textAlign:"right",color:RC[p.analystRating]||"#8899aa",fontSize:8}}>{p.analystRating||"—"}</td>
                        <td style={{padding:"5px 8px",textAlign:"right",color:"#4a9aff"}}>{p.annVol!=null?fmtPct(p.annVol):"—"}</td>
                      </tr>
                    ))}</tbody>
                  </table>
                </div>
                <div style={{background:"#060e1a",border:"1px solid #0c1824",borderRadius:8,padding:14}}>
                  <div style={{fontSize:9,color:"#1a3550",fontFamily:"monospace",marginBottom:8}}>YTD RETURN</div>
                  {peers.map((p,i)=>{
                    const v=p.ytdReturn||0,col=v>0?"#00e8a2":"#ff4060";
                    const mx=Math.max(...peers.map(x=>Math.abs(x.ytdReturn||0)),0.01);
                    return <div key={i} style={{marginBottom:6}}>
                      <div style={{display:"flex",justifyContent:"space-between",fontSize:8,fontFamily:"monospace",marginBottom:1}}>
                        <span style={{color:p.ticker===ticker?"#4a9aff":"#3a5070"}}>{p.ticker}</span>
                        <span style={{color:col}}>{v>0?"+":""}{(v*100).toFixed(1)}%</span>
                      </div>
                      <div style={{height:3,background:"#0a1520",borderRadius:2}}>
                        <div style={{height:"100%",width:`${Math.abs(v)/mx*100}%`,background:col,borderRadius:2}}/>
                      </div>
                    </div>;
                  })}
                </div>
              </div>;
            })()}
          </div>}

          {/* PORTFOLIO */}
          {tab==="portfolio"&&<div>
            <div style={{background:"#060e1a",border:"1px solid #0c1824",borderRadius:10,padding:18,marginBottom:12}}>
              <div style={{fontSize:10,color:"#1a3550",fontFamily:"monospace",marginBottom:12,letterSpacing:"0.1em"}}>PAPER TRADE · {ticker} · CURRENT PRICE: {fmtP(analysis.S0)}</div>
              <div style={{display:"flex",gap:8,alignItems:"flex-end",flexWrap:"wrap"}}>
                <div style={{flex:1,minWidth:110}}>
                  <div style={{fontSize:8,color:"#2a4060",fontFamily:"monospace",marginBottom:3}}>QUANTITY</div>
                  <input type="number" value={tradeQty} onChange={e=>setTradeQty(e.target.value)} min="1"
                    style={{width:"100%",background:"#0a1520",border:"1px solid #1a3050",borderRadius:6,padding:"8px 10px",color:"#d0dde8",fontFamily:"monospace",fontSize:14,outline:"none"}}/>
                </div>
                <div style={{flex:1,minWidth:110}}>
                  <div style={{fontSize:8,color:"#2a4060",fontFamily:"monospace",marginBottom:3}}>COST</div>
                  <div style={{background:"#0a1520",border:"1px solid #0c1824",borderRadius:6,padding:"8px 10px",color:"#8899aa",fontFamily:"monospace",fontSize:14}}>{fmtP((parseInt(tradeQty)||0)*analysis.S0)}</div>
                </div>
                <div style={{flex:1,minWidth:110}}>
                  <div style={{fontSize:8,color:"#2a4060",fontFamily:"monospace",marginBottom:3}}>POSITION</div>
                  <div style={{background:"#0a1520",border:"1px solid #0c1824",borderRadius:6,padding:"8px 10px",color:"#4a9aff",fontFamily:"monospace",fontSize:14}}>{portfolio.positions[ticker]?.shares||0} shares</div>
                </div>
                <div style={{display:"flex",gap:6}}>
                  <button onClick={()=>executeTrade("buy")} style={{padding:"9px 22px",background:"#002a10",border:"1px solid #00e8a2",borderRadius:6,color:"#00e8a2",fontFamily:"monospace",fontSize:13,fontWeight:700,cursor:"pointer"}}>▲ BUY</button>
                  <button onClick={()=>executeTrade("sell")} style={{padding:"9px 22px",background:"#2a0010",border:"1px solid #ff4060",borderRadius:6,color:"#ff4060",fontFamily:"monospace",fontSize:13,fontWeight:700,cursor:"pointer"}}>▼ SELL</button>
                </div>
              </div>
              {tradeMsg&&<div style={{marginTop:8,padding:"7px 10px",borderRadius:5,background:tradeMsg.type==="success"?"#002a1088":"#2a001088",border:`1px solid ${tradeMsg.type==="success"?"#00e8a2":"#ff4060"}33`,color:tradeMsg.type==="success"?"#00e8a2":"#ff4060",fontFamily:"monospace",fontSize:10}}>
                {tradeMsg.type==="success"?"✓":"✕"} {tradeMsg.msg}
              </div>}
            </div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:5,marginBottom:10}}>
              <Chip label="Total Value" value={`$${totalVal.toFixed(0)}`} color={colV(totalPnL)}/>
              <Chip label="Cash" value={`$${portfolio.cash.toFixed(0)}`} color="#4a9aff" sub="available"/>
              <Chip label="P&L" value={`${totalPnL>=0?"+":""}$${totalPnL.toFixed(0)}`} color={colV(totalPnL)} sub={fmtPct(totalPnL/100000)}/>
              <Chip label="Positions" value={Object.keys(portfolio.positions).length} color="#f0c040"/>
            </div>
            {Object.keys(portfolio.positions).length>0&&<div style={{background:"#060e1a",border:"1px solid #0c1824",borderRadius:8,padding:14,marginBottom:10}}>
              <div style={{fontSize:9,color:"#1a3550",fontFamily:"monospace",marginBottom:8}}>OPEN POSITIONS</div>
              <table style={{width:"100%",borderCollapse:"collapse",fontFamily:"monospace",fontSize:9}}>
                <thead><tr style={{borderBottom:"1px solid #0c1824"}}>
                  {["Ticker","Shares","Avg Cost","Current","Mkt Value","Unrealized P&L"].map(h=><th key={h} style={{padding:"4px 8px",color:"#1a3550",textAlign:"right",fontSize:8}}>{h}</th>)}
                </tr></thead>
                <tbody>{Object.entries(portfolio.positions).map(([sym,pos])=>{
                  const curr=sym===ticker?analysis.S0:(STOCK_META[sym]?.price||pos.avgCost);
                  const mv=curr*pos.shares,pnl2=mv-pos.totalCost;
                  return <tr key={sym} style={{borderBottom:"1px solid #080f1a"}}>
                    <td style={{padding:"4px 8px",color:"#4a9aff",fontWeight:700}}>{sym}</td>
                    <td style={{padding:"4px 8px",textAlign:"right",color:"#8899aa"}}>{pos.shares}</td>
                    <td style={{padding:"4px 8px",textAlign:"right",color:"#8899aa"}}>{fmtP(pos.avgCost)}</td>
                    <td style={{padding:"4px 8px",textAlign:"right",color:"#d0dde8"}}>{fmtP(curr)}</td>
                    <td style={{padding:"4px 8px",textAlign:"right",color:"#d0dde8"}}>{fmtP(mv)}</td>
                    <td style={{padding:"4px 8px",textAlign:"right",color:colV(pnl2)}}>{pnl2>=0?"+":""}{fmtP(pnl2)} ({fmtPct(pnl2/Math.max(pos.totalCost,1))})</td>
                  </tr>;
                })}</tbody>
              </table>
            </div>}
            {portfolio.history.length>0&&<div style={{background:"#060e1a",border:"1px solid #0c1824",borderRadius:8,padding:14}}>
              <div style={{fontSize:9,color:"#1a3550",fontFamily:"monospace",marginBottom:8}}>TRADE HISTORY ({portfolio.history.length})</div>
              <div style={{maxHeight:160,overflowY:"auto"}}>
                {[...portfolio.history].reverse().map((t,i)=>(
                  <div key={i} style={{display:"flex",justifyContent:"space-between",padding:"4px 0",borderBottom:"1px solid #0a1420",fontSize:9,fontFamily:"monospace"}}>
                    <span style={{color:t.type==="buy"?"#00e8a2":"#ff4060"}}>{t.type==="buy"?"▲":"▼"} {t.type.toUpperCase()}</span>
                    <span style={{color:"#4a9aff"}}>{t.ticker}</span>
                    <span style={{color:"#8899aa"}}>{t.qty} × {fmtP(t.price)}</span>
                    <span style={{color:"#5a7090"}}>{fmtP(t.qty*t.price)}</span>
                    <span style={{color:"#2a4060"}}>{t.time}</span>
                  </div>
                ))}
              </div>
            </div>}
            {portfolio.history.length===0&&Object.keys(portfolio.positions).length===0&&<div style={{padding:"24px 0",textAlign:"center",color:"#1a3050",fontFamily:"monospace",fontSize:10}}>
              No trades yet · Starting capital: $100,000 paper money
            </div>}
          </div>}

          <div style={{marginTop:18,paddingTop:8,borderTop:"1px solid #08121c",fontSize:7,color:"#0a1e30",fontFamily:"monospace",lineHeight:1.8}}>
            DISCLAIMER: Paper trading only — no real money involved. Prices via Claude Haiku web search; history simulated via GBM anchored to live price & IV. Models: GARCH(1,1) · 2-State HMM (Baum-Welch) · {analysis.paths}-path Monte Carlo · Black-Scholes (Δ Γ Θ ν ρ) · RSI · MACD · Bollinger · EMA Backtest · Sharpe · Sortino · Calmar · CVaR95. Not financial advice.
          </div>
        </div>}

        {status.phase==="idle"&&<div style={{padding:"50px 0",textAlign:"center",color:"#1a3050",fontFamily:"monospace",fontSize:11}}>Select a ticker to begin</div>}
      </div>
    </div>
  );
}
