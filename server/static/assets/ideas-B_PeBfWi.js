import{c as e,i as t,l as n,s as r,t as i}from"./colors--mnCBYbM.js";var a=40,o=class{feedEl;knowledgeEl;knowledgeEmptyEl;feedItems=[];statsEl;hypothesisCount=0;succeededCount=0;failedCount=0;messageCount=0;init(e){e.innerHTML=`
      <div class="ideas-page">
        <div class="ideas-header">
          <div class="ideas-title">
            <span class="stats-diamond">&#9670;</span>
            <span class="ideas-title-text">Collective Intelligence</span>
          </div>
          <div class="ideas-nav">
            <a href="/" class="ideas-nav-link">Dashboard</a>
            <span class="ideas-nav-active">Ideas</span>
          </div>
        </div>

        <div class="ideas-body">
          <div class="ideas-feed-col">
            <div class="ideas-col-label">RESEARCH FEED</div>
            <div class="ideas-feed" id="ideas-feed"></div>
          </div>
          <div class="ideas-knowledge-col">
            <div class="ideas-col-label">KNOWLEDGE STATE</div>
            <div class="ideas-knowledge" id="ideas-knowledge">
              <div class="ideas-knowledge-empty" id="ideas-knowledge-empty">
                <div class="knowledge-empty-icon">&#9671;</div>
                <div class="knowledge-empty-text">The curator agent will synthesize findings here as the swarm works...</div>
              </div>
            </div>
          </div>
        </div>

        <div class="ideas-stats" id="ideas-stats"></div>
      </div>
    `,this.feedEl=document.getElementById(`ideas-feed`),this.knowledgeEl=document.getElementById(`ideas-knowledge`),this.knowledgeEmptyEl=document.getElementById(`ideas-knowledge-empty`),this.statsEl=document.getElementById(`ideas-stats`)}handleMessage(e){if(e.type===`reset`){this.feedItems.forEach(e=>e.remove()),this.feedItems=[],this.feedEl.innerHTML=``,this.hypothesisCount=0,this.succeededCount=0,this.failedCount=0,this.messageCount=0,this.knowledgeEl.innerHTML=`
        <div class="ideas-knowledge-empty" id="ideas-knowledge-empty">
          <div class="knowledge-empty-icon">&#9671;</div>
          <div class="knowledge-empty-text">The curator agent will synthesize findings here as the swarm works...</div>
        </div>
      `,this.knowledgeEmptyEl=document.getElementById(`ideas-knowledge-empty`),this.updateStats();return}switch(e.type){case`chat_message`:this.addFeedItem({id:e.message_id,agentName:e.agent_name,agentId:e.agent_id||``,content:e.content,msgType:e.msg_type,timestamp:e.timestamp}),this.messageCount++;break;case`hypothesis_proposed`:this.hypothesisCount++,this.addFeedItem({id:e.hypothesis_id,agentName:e.agent_name,agentId:e.agent_id,content:`Proposed: "${e.title}"`,msgType:`agent`,timestamp:e.timestamp});break;case`hypothesis_status_changed`:e.new_status===`succeeded`&&this.succeededCount++,e.new_status===`failed`&&this.failedCount++;break;case`experiment_published`:if(e.is_new_best){let t=-e.improvement_pct,n=t>=0?`+`:``;this.addFeedItem({id:e.experiment_id,agentName:e.agent_name,agentId:e.agent_id,content:`NEW BEST: Score ${e.score.toFixed(0)} (${n}${t.toFixed(1)}%)`,msgType:`milestone`,timestamp:e.timestamp})}break;case`knowledge_updated`:this.renderKnowledge(e.content,e.updated_by);break}this.updateStats()}addFeedItem(e){let n=document.createElement(`div`);n.className=`feed-post feed-post--${e.msgType}`;let r=i(e.agentId||e.agentName),o=t(e.timestamp);for(e.msgType===`synthesis`?n.innerHTML=`
        <div class="feed-post-header">
          <span class="feed-post-badge synthesis-badge">SYNTHESIS</span>
          <span class="feed-post-time">${o}</span>
        </div>
        <div class="feed-post-content synthesis-content">${this.renderMarkdown(e.content)}</div>
        <div class="feed-post-author">— ${e.agentName}</div>
      `:e.msgType===`milestone`?n.innerHTML=`
        <div class="feed-post-header">
          <span class="feed-post-badge milestone-badge">&#9733; MILESTONE</span>
          <span class="feed-post-time">${o}</span>
        </div>
        <div class="feed-post-content milestone-content">${e.content}</div>
        <div class="feed-post-author">
          <span class="feed-post-dot" style="background:${r}"></span>
          ${e.agentName}
        </div>
      `:n.innerHTML=`
        <div class="feed-post-agent">
          <span class="feed-post-dot" style="background:${r}"></span>
          <span class="feed-post-name">${e.agentName}</span>
          <span class="feed-post-time">${o}</span>
        </div>
        <div class="feed-post-content">${e.content}</div>
      `,n.style.opacity=`0`,n.style.transform=`translateY(-16px)`,this.feedEl.prepend(n),requestAnimationFrame(()=>{n.style.transition=`opacity 0.35s ease, transform 0.35s ease`,n.style.opacity=`1`,n.style.transform=`translateY(0)`}),this.feedItems.unshift(n);this.feedItems.length>a;)this.feedItems.pop().remove()}renderKnowledge(e,n){this.knowledgeEmptyEl.style.display=`none`;let r=this.renderMarkdown(e);this.knowledgeEl.innerHTML=`
      <div class="knowledge-doc">${r}</div>
      <div class="knowledge-meta">Updated by ${n} at ${t(new Date().toISOString())}</div>
    `,this.knowledgeEl.style.boxShadow=`inset 0 0 30px rgba(0, 229, 255, 0.05)`,setTimeout(()=>{this.knowledgeEl.style.boxShadow=``},2e3)}renderMarkdown(e){return e.replace(/&/g,`&amp;`).replace(/</g,`&lt;`).replace(/>/g,`&gt;`).replace(/^## (.+)$/gm,`<h3 class="knowledge-h2">$1</h3>`).replace(/^### (.+)$/gm,`<h4 class="knowledge-h3">$1</h4>`).replace(/\*\*(.+?)\*\*/g,`<strong>$1</strong>`).replace(/^- (.+)$/gm,`<div class="knowledge-bullet">$1</div>`).replace(/\n\n/g,`<div class="knowledge-gap"></div>`).replace(/\n/g,`<br>`)}updateStats(){let e=this.hypothesisCount-this.succeededCount-this.failedCount;this.statsEl.innerHTML=`
      <span class="ideas-stat">HYPOTHESES <b>${this.hypothesisCount}</b></span>
      <span class="ideas-stat">SUCCEEDED <b style="color:var(--green)">${this.succeededCount}</b></span>
      <span class="ideas-stat">FAILED <b style="color:var(--red)">${this.failedCount}</b></span>
      <span class="ideas-stat">ACTIVE <b style="color:var(--cyan)">${Math.max(0,e)}</b></span>
      <span class="ideas-stat">MESSAGES <b>${this.messageCount}</b></span>
    `}},s=new URLSearchParams(window.location.search),c=s.has(`mock`),l=window.location.protocol===`https:`?`wss:`:`ws:`,u=s.get(`ws`)||`${l}//${window.location.host}/ws/dashboard`;function d(){return s.get(`api`)||u.replace(`ws://`,`http://`).replace(`wss://`,`https://`).replace(`/ws/dashboard`,``)}n(document.getElementById(`particleCanvas`));var f=document.getElementById(`ideas-root`),p=new o;p.init(f);function m(e){p.handleMessage(e)}document.addEventListener(`keydown`,e=>{e.key===`1`&&(window.location.href=`/`)});async function h(e){try{let t=await fetch(`${e}/api/state`);if(!t.ok)return;let n=await t.json(),r=[...n.active_hypotheses||[],...n.failed_hypotheses||[],...n.succeeded_hypotheses||[]];for(let e of r)m({type:`hypothesis_proposed`,hypothesis_id:e.id,agent_name:e.agent_name,agent_id:e.agent_id||``,title:e.title,description:e.description||``,strategy_tag:e.strategy_tag,parent_hypothesis_id:e.parent_hypothesis_id||null,timestamp:new Date().toISOString()}),(e.status===`succeeded`||e.status===`failed`)&&m({type:`hypothesis_status_changed`,hypothesis_id:e.id,new_status:e.status,agent_name:e.agent_name,timestamp:new Date().toISOString()});console.log(`[Ideas] Loaded ${r.length} hypotheses`);let[i,a]=await Promise.all([fetch(`${e}/api/messages?limit=50`),fetch(`${e}/api/knowledge`)]);if(i.ok){let e=await i.json();for(let t of e.reverse())m({type:`chat_message`,message_id:t.id,agent_name:t.agent_name,agent_id:t.agent_id,content:t.content,msg_type:t.msg_type,timestamp:t.created_at})}if(a.ok){let e=await a.json();e.content&&m({type:`knowledge_updated`,content:e.content,updated_by:e.updated_by,timestamp:e.updated_at})}}catch(e){console.warn(`[Ideas] Failed to load initial state:`,e)}}if(c){console.log(`[Ideas] Running in MOCK mode`);let e=new r;e.onMessage(m),e.start()}else{let t=d();console.log(`[Ideas] Connecting to ${u}, API: ${t}`),setTimeout(()=>h(t),300);let n=new e(u);n.onMessage(m),n.connect()}