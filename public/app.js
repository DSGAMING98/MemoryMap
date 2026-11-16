const STORAGE_KEY = "memorymap-vercel-v1";
const TITLES = {overview:"Overview",study:"Study session",predict:"Retention lab",schedule:"Revision plan",quiz:"Recall check",analytics:"Analytics",research:"Research report"};

const blankState = () => ({profile:{name:"Local learner"},sessions:[],predictions:[],quizzes:[]});
let state = loadState();

function loadState(){
  try{return {...blankState(),...JSON.parse(localStorage.getItem(STORAGE_KEY)||"null")};}
  catch{return blankState();}
}
function saveState(){localStorage.setItem(STORAGE_KEY,JSON.stringify(state));renderAll();}
function id(prefix){return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2,8)}`;}
function esc(value){return String(value??"").replace(/[&<>'"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));}
function pct(value,digits=1){return Number.isFinite(value)?`${(value*100).toFixed(digits)}%`:"—";}
function pp(value,digits=1){return Number.isFinite(value)?`${(value*100).toFixed(digits)} pp`:"—";}
function formatDate(value){return new Intl.DateTimeFormat(undefined,{dateStyle:"medium",timeStyle:"short"}).format(new Date(value));}
function daysBetween(start,end=Date.now()){return Math.max(0,(new Date(end)-new Date(start))/86400000);}
function fixedRetention(days,halfLife=1){return Math.max(0,Math.min(1,2**(-days/halfLife)));}
function intervalAtThreshold(halfLife=1,threshold=.7){return Math.max(1/24,Math.min(30,-halfLife*Math.log2(threshold)));}
function toast(message){const el=document.querySelector("#toast");el.textContent=message;el.classList.add("show");clearTimeout(toast.timer);toast.timer=setTimeout(()=>el.classList.remove("show"),3200);}

function showView(name){
  document.querySelectorAll(".view").forEach(v=>v.classList.toggle("active",v.id===name));
  document.querySelectorAll(".nav-item").forEach(v=>v.classList.toggle("active",v.dataset.view===name));
  document.querySelector("#viewTitle").textContent=TITLES[name]||"MemoryMap";
  document.querySelector(".sidebar").classList.remove("open");
  history.replaceState(null,"",`#${name}`);
  if(name==="analytics")renderAnalytics();
  if(name==="schedule")renderSchedule();
  window.scrollTo({top:0,behavior:"smooth"});
}
document.querySelectorAll("[data-view]").forEach(el=>el.addEventListener("click",()=>showView(el.dataset.view)));
document.querySelectorAll("[data-jump]").forEach(el=>el.addEventListener("click",()=>showView(el.dataset.jump)));
document.querySelector("#menuButton").addEventListener("click",()=>document.querySelector(".sidebar").classList.toggle("open"));

async function apiPrediction(session,days,halfLife=1,threshold=.7){
  const payload={delta_days:Number(days),history_seen:Number(session.historySeen||0),history_accuracy:Number(session.historyAccuracy||0),half_life_days:Number(halfLife),threshold:Number(threshold)};
  const response=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  if(!response.ok)throw new Error(`Prediction API returned ${response.status}`);
  return response.json();
}

async function checkApi(){
  const label=document.querySelector("#apiLabel"),pulse=document.querySelector(".pulse");
  try{const r=await fetch("/api/predict");if(!r.ok)throw new Error();const data=await r.json();label.textContent=`${data.model} ready`;pulse.classList.add("ready");}
  catch{label.textContent="Baseline mode available";pulse.classList.remove("ready");}
}

document.querySelector("#studyForm").addEventListener("submit",async event=>{
  event.preventDefault();
  const session={id:id("session"),learnerName:document.querySelector("#learnerName").value.trim(),topic:document.querySelector("#topicName").value.trim(),subject:document.querySelector("#subjectName").value.trim(),duration:Number(document.querySelector("#studyDuration").value),difficulty:Number(document.querySelector("#difficulty").value),confidence:Number(document.querySelector("#confidence").value),historySeen:Number(document.querySelector("#historySeen").value),historyAccuracy:Number(document.querySelector("#historyAccuracy").value)/100,notes:document.querySelector("#studyNotes").value.trim(),studiedAt:new Date().toISOString()};
  state.profile.name=session.learnerName||"Local learner";state.sessions.unshift(session);saveState();
  try{
    const result=await apiPrediction(session,1,1,.7);
    state.predictions.unshift({id:id("prediction"),sessionId:session.id,horizonDays:1,halfLife:1,threshold:.7,model:result.model_prediction,fixed:result.fixed_prediction,intervalDays:result.revision_interval_days,source:result.model,createdAt:new Date().toISOString()});
    saveState();toast("Session saved and prospective prediction recorded.");
  }catch{
    state.predictions.unshift({id:id("prediction"),sessionId:session.id,horizonDays:1,halfLife:1,threshold:.7,model:null,fixed:fixedRetention(1,1),intervalDays:intervalAtThreshold(1,.7),source:"Fixed fallback",createdAt:new Date().toISOString()});
    saveState();toast("Session saved. The fixed fallback was recorded.");
  }
  event.target.reset();document.querySelector("#studyDuration").value=30;document.querySelector("#historyAccuracy").value=70;
});

document.querySelector("#deltaDays").addEventListener("input",e=>document.querySelector("#deltaOutput").textContent=`${Number(e.target.value)} day${Number(e.target.value)===1?"":"s"}`);
document.querySelector("#threshold").addEventListener("input",e=>document.querySelector("#thresholdOutput").textContent=`${Math.round(Number(e.target.value)*100)}%`);

document.querySelector("#predictionForm").addEventListener("submit",async event=>{
  event.preventDefault();
  const session=state.sessions.find(s=>s.id===document.querySelector("#predictionSession").value);if(!session){toast("Record a study session first.");return;}
  const days=Number(document.querySelector("#deltaDays").value),halfLife=Number(document.querySelector("#halfLife").value),threshold=Number(document.querySelector("#threshold").value);
  let result;
  try{result=await apiPrediction(session,days,halfLife,threshold);}
  catch{result={model_prediction:null,fixed_prediction:fixedRetention(days,halfLife),revision_interval_days:intervalAtThreshold(halfLife,threshold),model:"Fixed fallback"};}
  const prediction={id:id("prediction"),sessionId:session.id,horizonDays:days,halfLife,threshold,model:result.model_prediction,fixed:result.fixed_prediction,intervalDays:result.revision_interval_days,source:result.model,createdAt:new Date().toISOString()};
  state.predictions.unshift(prediction);saveState();renderPredictionResult(session,prediction);renderCurve(halfLife,threshold);toast("Prospective prediction saved.");
});

document.querySelector("#quizForm").addEventListener("submit",event=>{
  event.preventDefault();const session=state.sessions.find(s=>s.id===document.querySelector("#quizSession").value);if(!session){toast("Record a study session first.");return;}
  const total=Number(document.querySelector("#totalQuestions").value),correct=Number(document.querySelector("#correctAnswers").value);if(correct>total){toast("Correct answers cannot exceed total questions.");return;}
  const delay=daysBetween(session.studiedAt),predictions=state.predictions.filter(p=>p.sessionId===session.id),matched=predictions.sort((a,b)=>Math.abs(a.horizonDays-delay)-Math.abs(b.horizonDays-delay))[0];
  state.quizzes.unshift({id:id("quiz"),sessionId:session.id,total,correct,recall:correct/total,confidence:Number(document.querySelector("#quizConfidence").value),delayDays:delay,predictionId:matched?.id||null,completedAt:new Date().toISOString()});saveState();toast("Recall result saved and matched to the nearest prospective horizon.");
});

function renderProfile(){const name=state.profile.name||"Local learner";document.querySelector("#profileName").textContent=name;document.querySelector("#profileInitial").textContent=name.trim().charAt(0).toUpperCase()||"L";document.querySelector("#learnerName").value=name==="Local learner"?"":name;}
function renderMetrics(){
  const topics=new Set(state.sessions.map(s=>s.topic.toLowerCase())).size,recall=state.quizzes.length?state.quizzes.reduce((n,q)=>n+q.recall,0)/state.quizzes.length:null,due=countDue();
  document.querySelector("#metricTopics").textContent=topics;document.querySelector("#metricSessions").textContent=state.sessions.length;document.querySelector("#metricRecall").textContent=pct(recall);document.querySelector("#metricDue").textContent=due;
}
function countDue(){return state.sessions.filter(s=>{const p=latestPrediction(s.id);return p&&Date.now()>=new Date(s.studiedAt).getTime()+p.intervalDays*86400000;}).length;}
function latestPrediction(sessionId){return state.predictions.filter(p=>p.sessionId===sessionId).sort((a,b)=>new Date(b.createdAt)-new Date(a.createdAt))[0];}
function renderOverviewQueue(){
  const el=document.querySelector("#overviewQueue"),items=state.sessions.slice(0,4);if(!items.length){el.className="empty-state";el.textContent="Your queue will appear after a study session.";return;}
  el.className="table-wrap";el.innerHTML=`<table><thead><tr><th>Topic</th><th>Recommended review</th><th>State</th></tr></thead><tbody>${items.map(s=>{const p=latestPrediction(s.id),due=p?new Date(new Date(s.studiedAt).getTime()+p.intervalDays*86400000):null,isDue=due&&due<=new Date();return `<tr><td><strong>${esc(s.topic)}</strong><br>${esc(s.subject||"General")}</td><td>${due?formatDate(due):"Awaiting prediction"}</td><td><span class="priority ${isDue?"high":"low"}">${isDue?"Due":"Scheduled"}</span></td></tr>`;}).join("")}</tbody></table>`;
}
function renderSessions(){
  const el=document.querySelector("#sessionTable");if(!state.sessions.length){el.innerHTML='<div class="empty-state">No study sessions recorded yet.</div>';return;}
  el.innerHTML=`<div class="table-wrap"><table><thead><tr><th>Topic</th><th>Studied</th><th>Duration</th><th>Difficulty</th><th>Confidence</th></tr></thead><tbody>${state.sessions.map(s=>`<tr><td><strong>${esc(s.topic)}</strong><br>${esc(s.subject||"General")}</td><td>${formatDate(s.studiedAt)}</td><td>${s.duration} min</td><td>${s.difficulty}/5</td><td>${s.confidence}/5</td></tr>`).join("")}</tbody></table></div>`;
}
function renderSelects(){
  const options=state.sessions.length?state.sessions.map(s=>`<option value="${s.id}">${esc(s.topic)} · ${formatDate(s.studiedAt)}</option>`).join(""):'<option value="">No sessions yet</option>';
  ["#predictionSession","#quizSession"].forEach(sel=>{const current=document.querySelector(sel).value;document.querySelector(sel).innerHTML=options;if(current&&state.sessions.some(s=>s.id===current))document.querySelector(sel).value=current;});
}
function renderPredictionResult(session,p){
  const el=document.querySelector("#predictionResult");if(!session||!p){el.innerHTML='<div class="empty-state">Select a saved study session to begin.</div>';return;}
  const due=new Date(new Date(session.studiedAt).getTime()+p.intervalDays*86400000);
  el.innerHTML=`<p class="eyebrow">${esc(session.topic).toUpperCase()}</p><h3>${Number(p.horizonDays)} day retention forecast</h3><div class="result-hero"><div class="score model"><span>Random Forest</span><strong>${p.model===null?"Unavailable":pct(p.model)}</strong></div><div class="score fixed"><span>Fixed curve</span><strong>${pct(p.fixed)}</strong></div></div><p class="explanation">Recommended review at a ${Math.round(p.threshold*100)}% baseline threshold: <strong>${formatDate(due)}</strong>. Source: ${esc(p.source)}. The public model is a bootstrap estimate, not a learner-specific clinical judgment.</p>`;
}
function renderCurve(halfLife=1,threshold=.7){
  const w=900,h=250,pad=35,points=Array.from({length:61},(_,i)=>({d:i/2,r:fixedRetention(i/2,halfLife)}));
  const line=points.map((p,i)=>`${i?"L":"M"} ${pad+p.d/30*(w-pad*2)} ${h-pad-p.r*(h-pad*2)}`).join(" "),ty=h-pad-threshold*(h-pad*2);
  document.querySelector("#curveChart").innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Thirty-day forgetting curve"><line class="axis" x1="${pad}" y1="${h-pad}" x2="${w-pad}" y2="${h-pad}"/><line class="axis" x1="${pad}" y1="${pad}" x2="${pad}" y2="${h-pad}"/><line class="threshold-line" x1="${pad}" y1="${ty}" x2="${w-pad}" y2="${ty}"/><path class="curve" d="${line}"/><text class="chart-label" x="${pad}" y="${h-8}">0 days</text><text class="chart-label" x="${w-pad-42}" y="${h-8}">30 days</text><text class="chart-label" x="${pad+8}" y="${ty-7}">${Math.round(threshold*100)}% threshold</text></svg>`;
}
function renderSchedule(){
  const el=document.querySelector("#scheduleList");if(!state.sessions.length){el.innerHTML='<div class="card empty-state">Record a study session to generate a revision plan.</div>';return;}
  el.innerHTML=state.sessions.map(s=>{const p=latestPrediction(s.id)||{intervalDays:intervalAtThreshold(1,.7),threshold:.7,source:"Fixed fallback"},due=new Date(new Date(s.studiedAt).getTime()+p.intervalDays*86400000),remaining=(due-Date.now())/86400000,level=remaining<=0?"high":remaining<=1?"medium":"low",label=remaining<=0?"Due now":remaining<=1?"Due soon":"Scheduled";return `<article class="card schedule-card"><div><h3>${esc(s.topic)}</h3><p>${esc(s.subject||"General")} · studied ${formatDate(s.studiedAt)}</p><div class="schedule-meta"><span class="chip">${Math.round(p.threshold*100)}% threshold</span><span class="chip">${esc(p.source)}</span><span class="chip">${p.intervalDays.toFixed(2)} days</span></div></div><div><span class="priority ${level}">${label}</span><p>${formatDate(due)}</p></div></article>`;}).join("");
}
document.querySelector("#refreshSchedule").addEventListener("click",()=>{renderSchedule();toast("Priorities refreshed.");});
function renderQuizzes(){
  const el=document.querySelector("#quizHistory");if(!state.quizzes.length){el.innerHTML='<div class="empty-state">No recall checks have been recorded.</div>';return;}
  el.innerHTML=`<div class="table-wrap"><table><thead><tr><th>Topic</th><th>Delay</th><th>Actual</th><th>Predicted</th><th>Error</th></tr></thead><tbody>${state.quizzes.map(q=>{const s=state.sessions.find(s=>s.id===q.sessionId),p=state.predictions.find(p=>p.id===q.predictionId),pred=p?(p.model??p.fixed):null,error=pred===null?null:Math.abs(pred-q.recall);return `<tr><td><strong>${esc(s?.topic||"Unknown")}</strong></td><td>${q.delayDays.toFixed(2)} days</td><td>${pct(q.recall)}</td><td>${pct(pred)}</td><td>${pp(error)}</td></tr>`;}).join("")}</tbody></table></div>`;
}
function renderAnalytics(){
  const minutes=state.sessions.reduce((n,s)=>n+s.duration,0),confidence=state.sessions.length?state.sessions.reduce((n,s)=>n+s.confidence,0)/state.sessions.length:null,matched=state.quizzes.map(q=>{const p=state.predictions.find(p=>p.id===q.predictionId);return p?Math.abs((p.model??p.fixed)-q.recall):null;}).filter(Number.isFinite),mae=matched.length?matched.reduce((a,b)=>a+b,0)/matched.length:null;
  document.querySelector("#analyticsMinutes").textContent=`${minutes}m`;document.querySelector("#analyticsQuizzes").textContent=state.quizzes.length;document.querySelector("#analyticsMae").textContent=pp(mae);document.querySelector("#analyticsConfidence").textContent=confidence?`${confidence.toFixed(1)}/5`:"—";
  const topics={};state.quizzes.forEach(q=>{const s=state.sessions.find(s=>s.id===q.sessionId),name=s?.topic||"Unknown";(topics[name]??=[]).push(q.recall);});renderBars("#topicBars",Object.entries(topics).map(([name,values])=>({name,value:values.reduce((a,b)=>a+b,0)/values.length,label:pct(values.reduce((a,b)=>a+b,0)/values.length)})),"No topic recall evidence yet.");
  const errors=state.quizzes.map(q=>{const s=state.sessions.find(s=>s.id===q.sessionId),p=state.predictions.find(p=>p.id===q.predictionId),value=p?Math.abs((p.model??p.fixed)-q.recall):null;return Number.isFinite(value)?{name:s?.topic||"Unknown",value,label:pp(value)}:null;}).filter(Boolean);renderBars("#errorBars",errors,"No matched predictions yet.");
}
function renderBars(selector,items,empty){const el=document.querySelector(selector);if(!items.length){el.innerHTML=`<div class="empty-state">${empty}</div>`;return;}el.innerHTML=items.map(i=>`<div class="bar-row"><span>${esc(i.name)}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.min(100,i.value*100)}%"></div></div><strong>${i.label}</strong></div>`).join("");}
function renderAll(){renderProfile();renderMetrics();renderOverviewQueue();renderSessions();renderSelects();renderSchedule();renderQuizzes();renderAnalytics();const session=state.sessions[0],p=session&&latestPrediction(session.id);renderPredictionResult(session,p);}

renderCurve();renderAll();checkApi();showView(location.hash.slice(1) in TITLES?location.hash.slice(1):"overview");
