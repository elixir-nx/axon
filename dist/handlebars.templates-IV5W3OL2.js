(()=>{(function(){var d=Handlebars.template,y=Handlebars.templates=Handlebars.templates||{};y["autocomplete-suggestions"]=d({1:function(n,l,a,c,s){var e,o,u=l??(n.nullContext||{}),r=n.hooks.helperMissing,i="function",t=n.escapeExpression,f=n.lookupProperty||function(p,m){if(Object.prototype.hasOwnProperty.call(p,m))return p[m]};return'    <a href="'+t((o=(o=f(a,"link")||(l!=null?f(l,"link"):l))!=null?o:r,typeof o===i?o.call(u,{name:"link",hash:{},data:s,loc:{start:{line:7,column:13},end:{line:7,column:21}}}):o))+'" class="autocomplete-suggestion" data-index="'+t((o=(o=f(a,"index")||s&&f(s,"index"))!=null?o:r,typeof o===i?o.call(u,{name:"index",hash:{},data:s,loc:{start:{line:7,column:67},end:{line:7,column:77}}}):o))+`" tabindex="-1">
      <div class="title">
        <span translate="no">`+((e=(o=(o=f(a,"title")||(l!=null?f(l,"title"):l))!=null?o:r,typeof o===i?o.call(u,{name:"title",hash:{},data:s,loc:{start:{line:9,column:29},end:{line:9,column:40}}}):o))!=null?e:"")+`</span>
`+((e=f(a,"if").call(u,l!=null?f(l,"label"):l,{name:"if",hash:{},fn:n.program(2,s,0),inverse:n.noop,data:s,loc:{start:{line:10,column:8},end:{line:12,column:15}}}))!=null?e:"")+`      </div>

`+((e=f(a,"if").call(u,l!=null?f(l,"description"):l,{name:"if",hash:{},fn:n.program(4,s,0),inverse:n.noop,data:s,loc:{start:{line:15,column:6},end:{line:19,column:13}}}))!=null?e:"")+`    </a>
`},2:function(n,l,a,c,s){var e,o=n.lookupProperty||function(u,r){if(Object.prototype.hasOwnProperty.call(u,r))return u[r]};return'          <span class="label">('+n.escapeExpression((e=(e=o(a,"label")||(l!=null?o(l,"label"):l))!=null?e:n.hooks.helperMissing,typeof e=="function"?e.call(l??(n.nullContext||{}),{name:"label",hash:{},data:s,loc:{start:{line:11,column:31},end:{line:11,column:40}}}):e))+`)</span>
`},4:function(n,l,a,c,s){var e,o,u=n.lookupProperty||function(r,i){if(Object.prototype.hasOwnProperty.call(r,i))return r[i]};return`        <div class="description" translate="no">
          `+((e=(o=(o=u(a,"description")||(l!=null?u(l,"description"):l))!=null?o:n.hooks.helperMissing,typeof o=="function"?o.call(l??(n.nullContext||{}),{name:"description",hash:{},data:s,loc:{start:{line:17,column:10},end:{line:17,column:27}}}):o))!=null?e:"")+`
        </div>
`},compiler:[8,">= 4.3.0"],main:function(n,l,a,c,s){var e,o,u=l??(n.nullContext||{}),r=n.hooks.helperMissing,i="function",t=n.escapeExpression,f=n.lookupProperty||function(p,m){if(Object.prototype.hasOwnProperty.call(p,m))return p[m]};return`<div class="autocomplete-suggestions">
  <a class="autocomplete-suggestion" href="search.html?q=`+t((o=(o=f(a,"term")||(l!=null?f(l,"term"):l))!=null?o:r,typeof o===i?o.call(u,{name:"term",hash:{},data:s,loc:{start:{line:2,column:57},end:{line:2,column:65}}}):o))+`" data-index="-1" tabindex="-1">
    <div class="title">"<em>`+t((o=(o=f(a,"term")||(l!=null?f(l,"term"):l))!=null?o:r,typeof o===i?o.call(u,{name:"term",hash:{},data:s,loc:{start:{line:3,column:28},end:{line:3,column:36}}}):o))+`</em>"</div>
    <div class="description">Search the documentation</div>
  </a>
`+((e=f(a,"each").call(u,l!=null?f(l,"suggestions"):l,{name:"each",hash:{},fn:n.program(1,s,0),inverse:n.noop,data:s,loc:{start:{line:6,column:2},end:{line:21,column:11}}}))!=null?e:"")+`</div>
`},useData:!0}),y["modal-layout"]=d({compiler:[8,">= 4.3.0"],main:function(n,l,a,c,s){return`<div id="modal" class="modal" tabindex="-1">
  <div class="modal-contents">
    <div class="modal-header">
      <div class="modal-title"></div>
      <button class="modal-close">\xD7</button>
    </div>
    <div class="modal-body">
    </div>
  </div>
</div>
`},useData:!0}),y["quick-switch-modal-body"]=d({compiler:[8,">= 4.3.0"],main:function(n,l,a,c,s){return`<div id="quick-switch-modal-body">
  <i class="ri-search-2-line" aria-hidden="true"></i>
  <input type="text" id="quick-switch-input" class="search-input" placeholder="Jump to..." autocomplete="off" spellcheck="false">
  <div id="quick-switch-results"></div>
</div>
`},useData:!0}),y["quick-switch-results"]=d({1:function(n,l,a,c,s){var e,o=l??(n.nullContext||{}),u=n.hooks.helperMissing,r="function",i=n.escapeExpression,t=n.lookupProperty||function(f,p){if(Object.prototype.hasOwnProperty.call(f,p))return f[p]};return'  <div class="quick-switch-result" data-index="'+i((e=(e=t(a,"index")||s&&t(s,"index"))!=null?e:u,typeof e===r?e.call(o,{name:"index",hash:{},data:s,loc:{start:{line:2,column:47},end:{line:2,column:57}}}):e))+`">
    `+i((e=(e=t(a,"name")||(l!=null?t(l,"name"):l))!=null?e:u,typeof e===r?e.call(o,{name:"name",hash:{},data:s,loc:{start:{line:3,column:4},end:{line:3,column:12}}}):e))+`
  </div>
`},compiler:[8,">= 4.3.0"],main:function(n,l,a,c,s){var e,o=n.lookupProperty||function(u,r){if(Object.prototype.hasOwnProperty.call(u,r))return u[r]};return(e=o(a,"each").call(l??(n.nullContext||{}),l!=null?o(l,"results"):l,{name:"each",hash:{},fn:n.program(1,s,0),inverse:n.noop,data:s,loc:{start:{line:1,column:0},end:{line:5,column:9}}}))!=null?e:""},useData:!0}),y["search-results"]=d({1:function(n,l,a,c,s){var e,o=n.lookupProperty||function(u,r){if(Object.prototype.hasOwnProperty.call(u,r))return u[r]};return"    Search results for <em>"+n.escapeExpression((e=(e=o(a,"value")||(l!=null?o(l,"value"):l))!=null?e:n.hooks.helperMissing,typeof e=="function"?e.call(l??(n.nullContext||{}),{name:"value",hash:{},data:s,loc:{start:{line:3,column:27},end:{line:3,column:36}}}):e))+`</em>
`},3:function(n,l,a,c,s){return`    Invalid search
`},5:function(n,l,a,c,s){var e,o=n.lookupProperty||function(u,r){if(Object.prototype.hasOwnProperty.call(u,r))return u[r]};return(e=o(a,"each").call(l??(n.nullContext||{}),l!=null?o(l,"results"):l,{name:"each",hash:{},fn:n.program(6,s,0),inverse:n.noop,data:s,loc:{start:{line:15,column:2},end:{line:26,column:11}}}))!=null?e:""},6:function(n,l,a,c,s){var e,o=n.lambda,u=n.escapeExpression,r=n.lookupProperty||function(i,t){if(Object.prototype.hasOwnProperty.call(i,t))return i[t]};return`    <div class="result">
      <h2 class="result-id">
        <a href="`+u(o(l!=null?r(l,"ref"):l,l))+`">
          <span translate="no">`+u(o(l!=null?r(l,"title"):l,l))+"</span> <small>("+u(o(l!=null?r(l,"type"):l,l))+`)</small>
        </a>
      </h2>
`+((e=r(a,"each").call(l??(n.nullContext||{}),l!=null?r(l,"excerpts"):l,{name:"each",hash:{},fn:n.program(7,s,0),inverse:n.noop,data:s,loc:{start:{line:22,column:8},end:{line:24,column:17}}}))!=null?e:"")+`    </div>
`},7:function(n,l,a,c,s){var e;return'          <p class="result-elem">'+((e=n.lambda(l,l))!=null?e:"")+`</p>
`},9:function(n,l,a,c,s){var e,o=n.lookupProperty||function(u,r){if(Object.prototype.hasOwnProperty.call(u,r))return u[r]};return((e=(o(a,"isArray")||l&&o(l,"isArray")||n.hooks.helperMissing).call(l??(n.nullContext||{}),l!=null?o(l,"results"):l,{name:"isArray",hash:{},fn:n.program(10,s,0),inverse:n.program(12,s,0),data:s,loc:{start:{line:28,column:2},end:{line:34,column:14}}}))!=null?e:"")+`
  <p>The search functionality is full-text based. Here are some tips:</p>

  <ul>
    <li>Multiple words (such as <code>foo bar</code>) are searched as <code>OR</code></li>
    <li>Use <code>*</code> anywhere (such as <code>fo*</code>) as wildcard</li>
    <li>Use <code>+</code> before a word (such as <code>+foo</code>) to make its presence required</li>
    <li>Use <code>-</code> before a word (such as <code>-foo</code>) to make its absence required</li>
    <li>Use <code>:</code> to search on a particular field (such as <code>field:word</code>). The available fields are <code>title</code> and <code>doc</code></li>
    <li>Use <code>WORD^NUMBER</code> (such as <code>foo^2</code>) to boost the given word</li>
    <li>Use <code>WORD~NUMBER</code> (such as <code>foo~2</code>) to do a search with edit distance on word</li>
  </ul>

  <p>To quickly go to a module, type, or function, use the autocompletion feature in the sidebar search.</p>
`},10:function(n,l,a,c,s){var e,o=n.lookupProperty||function(u,r){if(Object.prototype.hasOwnProperty.call(u,r))return u[r]};return"    <p>Sorry, we couldn't find anything for <em>"+n.escapeExpression((e=(e=o(a,"value")||(l!=null?o(l,"value"):l))!=null?e:n.hooks.helperMissing,typeof e=="function"?e.call(l??(n.nullContext||{}),{name:"value",hash:{},data:s,loc:{start:{line:29,column:48},end:{line:29,column:57}}}):e))+`</em>.</p>
`},12:function(n,l,a,c,s){var e,o=n.lookupProperty||function(u,r){if(Object.prototype.hasOwnProperty.call(u,r))return u[r]};return(e=o(a,"if").call(l??(n.nullContext||{}),l!=null?o(l,"value"):l,{name:"if",hash:{},fn:n.program(13,s,0),inverse:n.program(15,s,0),data:s,loc:{start:{line:30,column:2},end:{line:34,column:2}}}))!=null?e:""},13:function(n,l,a,c,s){var e,o=n.lookupProperty||function(u,r){if(Object.prototype.hasOwnProperty.call(u,r))return u[r]};return"    <p>Invalid search: "+n.escapeExpression((e=(e=o(a,"errorMessage")||(l!=null?o(l,"errorMessage"):l))!=null?e:n.hooks.helperMissing,typeof e=="function"?e.call(l??(n.nullContext||{}),{name:"errorMessage",hash:{},data:s,loc:{start:{line:31,column:23},end:{line:31,column:39}}}):e))+`.</p>
`},15:function(n,l,a,c,s){return`    <p>Please type something into the search bar to perform a search.</p>
  `},compiler:[8,">= 4.3.0"],main:function(n,l,a,c,s){var e,o=l??(n.nullContext||{}),u=n.lookupProperty||function(r,i){if(Object.prototype.hasOwnProperty.call(r,i))return r[i]};return`<h1>
`+((e=u(a,"if").call(o,l!=null?u(l,"value"):l,{name:"if",hash:{},fn:n.program(1,s,0),inverse:n.program(3,s,0),data:s,loc:{start:{line:2,column:2},end:{line:6,column:9}}}))!=null?e:"")+`
  <button class="icon-action display-settings">
    <i class="ri-settings-3-line"></i>
    <span class="sr-only">Settings</span>
  </button>
</h1>

`+((e=(u(a,"isNonEmptyArray")||l&&u(l,"isNonEmptyArray")||n.hooks.helperMissing).call(o,l!=null?u(l,"results"):l,{name:"isNonEmptyArray",hash:{},fn:n.program(5,s,0),inverse:n.program(9,s,0),data:s,loc:{start:{line:14,column:0},end:{line:49,column:20}}}))!=null?e:"")},useData:!0}),y["settings-modal-body"]=d({1:function(n,l,a,c,s){var e,o=n.lookupProperty||function(u,r){if(Object.prototype.hasOwnProperty.call(u,r))return u[r]};return(e=o(a,"if").call(l??(n.nullContext||{}),l!=null?o(l,"description"):l,{name:"if",hash:{},fn:n.program(2,s,0),inverse:n.noop,data:s,loc:{start:{line:40,column:6},end:{line:53,column:13}}}))!=null?e:""},2:function(n,l,a,c,s){var e,o=n.lookupProperty||function(u,r){if(Object.prototype.hasOwnProperty.call(u,r))return u[r]};return`        <dl class="shortcut-row">
          <dd class="shortcut-description">
            `+n.escapeExpression(n.lambda(l!=null?o(l,"description"):l,l))+`
          </dd>
          <dt class="shortcut-keys">
`+((e=o(a,"if").call(l??(n.nullContext||{}),l!=null?o(l,"displayAs"):l,{name:"if",hash:{},fn:n.program(3,s,0),inverse:n.program(5,s,0),data:s,loc:{start:{line:46,column:12},end:{line:50,column:19}}}))!=null?e:"")+`          </dt>
        </dl>
`},3:function(n,l,a,c,s){var e,o=n.lookupProperty||function(u,r){if(Object.prototype.hasOwnProperty.call(u,r))return u[r]};return"              "+((e=n.lambda(l!=null?o(l,"displayAs"):l,l))!=null?e:"")+`
`},5:function(n,l,a,c,s){var e=n.lookupProperty||function(o,u){if(Object.prototype.hasOwnProperty.call(o,u))return o[u]};return"              <kbd><kbd>"+n.escapeExpression(n.lambda(l!=null?e(l,"key"):l,l))+`</kbd></kbd>
`},compiler:[8,">= 4.3.0"],main:function(n,l,a,c,s){var e,o=n.lookupProperty||function(u,r){if(Object.prototype.hasOwnProperty.call(u,r))return u[r]};return`<div id="settings-modal-content">
  <div id="settings-content">
    <label class="switch-button-container">
      <div>
        <span>Theme</span>
        <p>Use the documentation UI in a theme.</p>
      </div>
      <div>
        <select name="theme" class="settings-select">
          <option value="dark">Dark</option>
          <option value="light">Light</option>
          <option value="system">System</option>
        </select>
      </div>
    </label>
    <label class="switch-button-container">
      <div>
        <span>Show tooltips</span>
        <p>Show tooltips when mousing over code references.</p>
      </div>
      <div class="switch-button">
        <input class="switch-button__checkbox" type="checkbox" name="tooltips" />
        <div class="switch-button__bg"></div>
      </div>
    </label>
    <label class="switch-button-container">
      <div>
        <span>Run in Livebook</span>
        <p>Use Direct Address for \u201CRun in Livebook\u201D badges.</p>
      </div>
      <div class="switch-button">
        <input class="switch-button__checkbox" type="checkbox" name="direct_livebook_url" />
        <div class="switch-button__bg"></div>
      </div>
    </label>
    <input class="input" type="url" name="livebook_url" placeholder="Enter Livebook instance URL" aria-label="Enter Livebook instance URL" />
  </div>
  <div id="keyboard-shortcuts-content" class="hidden">
`+((e=o(a,"each").call(l??(n.nullContext||{}),l!=null?o(l,"shortcuts"):l,{name:"each",hash:{},fn:n.program(1,s,0),inverse:n.noop,data:s,loc:{start:{line:39,column:4},end:{line:54,column:13}}}))!=null?e:"")+`  </div>
</div>
`},useData:!0}),y["sidebar-items"]=d({1:function(n,l,a,c,s,e,o){var u,r=l??(n.nullContext||{}),i=n.hooks.helperMissing,t=n.lookupProperty||function(f,p){if(Object.prototype.hasOwnProperty.call(f,p))return f[p]};return((u=(t(a,"groupChanged")||l&&t(l,"groupChanged")||i).call(r,o[1],(u=e[0][0])!=null?t(u,"group"):u,{name:"groupChanged",hash:{},fn:n.program(2,s,0,e,o),inverse:n.noop,data:s,blockParams:e,loc:{start:{line:2,column:2},end:{line:6,column:19}}}))!=null?u:"")+`
`+((u=(t(a,"nestingChanged")||l&&t(l,"nestingChanged")||i).call(r,o[1],e[0][0],{name:"nestingChanged",hash:{},fn:n.program(7,s,0,e,o),inverse:n.noop,data:s,blockParams:e,loc:{start:{line:8,column:2},end:{line:10,column:21}}}))!=null?u:"")+`
  <li class="`+((u=(t(a,"isLocal")||l&&t(l,"isLocal")||i).call(r,(u=e[0][0])!=null?t(u,"id"):u,{name:"isLocal",hash:{},fn:n.program(9,s,0,e,o),inverse:n.noop,data:s,blockParams:e,loc:{start:{line:12,column:13},end:{line:12,column:62}}}))!=null?u:"")+`">
    <a href="`+n.escapeExpression(n.lambda((u=e[0][0])!=null?t(u,"id"):u,l))+".html"+((u=(t(a,"isLocal")||l&&t(l,"isLocal")||i).call(r,(u=e[0][0])!=null?t(u,"id"):u,{name:"isLocal",hash:{},fn:n.program(11,s,0,e,o),inverse:n.noop,data:s,blockParams:e,loc:{start:{line:13,column:29},end:{line:13,column:69}}}))!=null?u:"")+'" class="expand" '+((u=(t(a,"isArray")||l&&t(l,"isArray")||i).call(r,(u=e[0][0])!=null?t(u,"headers"):u,{name:"isArray",hash:{},fn:n.program(3,s,0,e,o),inverse:n.program(5,s,0,e,o),data:s,blockParams:e,loc:{start:{line:13,column:86},end:{line:13,column:145}}}))!=null?u:"")+`>
`+((u=t(a,"if").call(r,(u=e[0][0])!=null?t(u,"nested_title"):u,{name:"if",hash:{},fn:n.program(13,s,0,e,o),inverse:n.program(15,s,0,e,o),data:s,blockParams:e,loc:{start:{line:14,column:6},end:{line:18,column:13}}}))!=null?u:"")+`      <span class="icon-expand"></span>
    </a>

    <ul>
`+((u=(t(a,"isArray")||l&&t(l,"isArray")||i).call(r,(u=e[0][0])!=null?t(u,"headers"):u,{name:"isArray",hash:{},fn:n.program(17,s,0,e,o),inverse:n.program(20,s,0,e,o),data:s,blockParams:e,loc:{start:{line:23,column:6},end:{line:65,column:18}}}))!=null?u:"")+`      </ul>
  </li>
`},2:function(n,l,a,c,s,e){var o,u=n.lookupProperty||function(r,i){if(Object.prototype.hasOwnProperty.call(r,i))return r[i]};return'    <li class="group" '+((o=(u(a,"isArray")||l&&u(l,"isArray")||n.hooks.helperMissing).call(l??(n.nullContext||{}),(o=e[1][0])!=null?u(o,"headers"):o,{name:"isArray",hash:{},fn:n.program(3,s,0,e),inverse:n.program(5,s,0,e),data:s,blockParams:e,loc:{start:{line:3,column:22},end:{line:3,column:81}}}))!=null?o:"")+`>
      `+n.escapeExpression(n.lambda((o=e[1][0])!=null?u(o,"group"):o,l))+`
    </li>
`},3:function(n,l,a,c,s){return""},5:function(n,l,a,c,s){return'translate="no"'},7:function(n,l,a,c,s,e){var o,u=n.lookupProperty||function(r,i){if(Object.prototype.hasOwnProperty.call(r,i))return r[i]};return'    <li class="nesting-context" aria-hidden="true" translate="no">'+n.escapeExpression(n.lambda((o=e[1][0])!=null?u(o,"nested_context"):o,l))+`</li>
`},9:function(n,l,a,c,s){return"current-page open"},11:function(n,l,a,c,s){return"#content"},13:function(n,l,a,c,s,e){var o,u=n.lookupProperty||function(r,i){if(Object.prototype.hasOwnProperty.call(r,i))return r[i]};return"        "+((o=n.lambda((o=e[1][0])!=null?u(o,"nested_title"):o,l))!=null?o:"")+`
`},15:function(n,l,a,c,s,e){var o,u=n.lookupProperty||function(r,i){if(Object.prototype.hasOwnProperty.call(r,i))return r[i]};return"        "+((o=n.lambda((o=e[1][0])!=null?u(o,"title"):o,l))!=null?o:"")+`
`},17:function(n,l,a,c,s,e){var o,u=n.lookupProperty||function(r,i){if(Object.prototype.hasOwnProperty.call(r,i))return r[i]};return(o=u(a,"each").call(l??(n.nullContext||{}),(o=e[1][0])!=null?u(o,"headers"):o,{name:"each",hash:{},fn:n.program(18,s,0,e),inverse:n.noop,data:s,blockParams:e,loc:{start:{line:24,column:8},end:{line:28,column:17}}}))!=null?o:""},18:function(n,l,a,c,s,e){var o,u,r=l??(n.nullContext||{}),i=n.hooks.helperMissing,t="function",f=n.lookupProperty||function(p,m){if(Object.prototype.hasOwnProperty.call(p,m))return p[m]};return`          <li>
            <a href="`+n.escapeExpression(n.lambda((o=e[2][0])!=null?f(o,"id"):o,l))+".html#"+((o=(u=(u=f(a,"anchor")||(l!=null?f(l,"anchor"):l))!=null?u:i,typeof u===t?u.call(r,{name:"anchor",hash:{},data:s,blockParams:e,loc:{start:{line:26,column:38},end:{line:26,column:50}}}):u))!=null?o:"")+'">'+((o=(u=(u=f(a,"id")||(l!=null?f(l,"id"):l))!=null?u:i,typeof u===t?u.call(r,{name:"id",hash:{},data:s,blockParams:e,loc:{start:{line:26,column:52},end:{line:26,column:60}}}):u))!=null?o:"")+`</a>
          </li>
`},20:function(n,l,a,c,s,e){var o,u=l??(n.nullContext||{}),r=n.hooks.helperMissing,i=n.lookupProperty||function(t,f){if(Object.prototype.hasOwnProperty.call(t,f))return t[f]};return((o=(i(a,"showSections")||l&&i(l,"showSections")||r).call(u,e[1][0],{name:"showSections",hash:{},fn:n.program(21,s,0,e),inverse:n.noop,data:s,blockParams:e,loc:{start:{line:30,column:8},end:{line:44,column:25}}}))!=null?o:"")+((o=(i(a,"showSummary")||l&&i(l,"showSummary")||r).call(u,e[1][0],{name:"showSummary",hash:{},fn:n.program(26,s,0,e),inverse:n.noop,data:s,blockParams:e,loc:{start:{line:45,column:8},end:{line:49,column:24}}}))!=null?o:"")+((o=i(a,"each").call(u,(o=e[1][0])!=null?i(o,"nodeGroups"):o,{name:"each",hash:{},fn:n.program(28,s,1,e),inverse:n.noop,data:s,blockParams:e,loc:{start:{line:50,column:8},end:{line:64,column:17}}}))!=null?o:"")},21:function(n,l,a,c,s,e){var o,u=l??(n.nullContext||{}),r=n.lookupProperty||function(i,t){if(Object.prototype.hasOwnProperty.call(i,t))return i[t]};return'          <li class="docs '+((o=(r(a,"isLocal")||l&&r(l,"isLocal")||n.hooks.helperMissing).call(u,(o=e[2][0])!=null?r(o,"id"):o,{name:"isLocal",hash:{},fn:n.program(22,s,0,e),inverse:n.noop,data:s,blockParams:e,loc:{start:{line:31,column:26},end:{line:31,column:62}}}))!=null?o:"")+`">
            <a href="`+n.escapeExpression(n.lambda((o=e[2][0])!=null?r(o,"id"):o,l))+`.html#content" class="expand">
              Sections
              <span class="icon-expand"></span>
            </a>
            <ul class="sections-list deflist">
`+((o=r(a,"each").call(u,l!=null?r(l,"sections"):l,{name:"each",hash:{},fn:n.program(24,s,0,e),inverse:n.noop,data:s,blockParams:e,loc:{start:{line:37,column:14},end:{line:41,column:23}}}))!=null?o:"")+`            </ul>
          </li>
`},22:function(n,l,a,c,s){return"open"},24:function(n,l,a,c,s,e){var o,u,r=n.escapeExpression,i=l??(n.nullContext||{}),t=n.hooks.helperMissing,f="function",p=n.lookupProperty||function(m,v){if(Object.prototype.hasOwnProperty.call(m,v))return m[v]};return`                <li>
                  <a href="`+r(n.lambda((o=e[3][0])!=null?p(o,"id"):o,l))+".html#"+r((u=(u=p(a,"anchor")||(l!=null?p(l,"anchor"):l))!=null?u:t,typeof u===f?u.call(i,{name:"anchor",hash:{},data:s,blockParams:e,loc:{start:{line:39,column:44},end:{line:39,column:54}}}):u))+'">'+((o=(u=(u=p(a,"id")||(l!=null?p(l,"id"):l))!=null?u:t,typeof u===f?u.call(i,{name:"id",hash:{},data:s,blockParams:e,loc:{start:{line:39,column:56},end:{line:39,column:64}}}):u))!=null?o:"")+`</a>
                </li>
`},26:function(n,l,a,c,s,e){var o,u=n.lookupProperty||function(r,i){if(Object.prototype.hasOwnProperty.call(r,i))return r[i]};return`          <li>
            <a href="`+n.escapeExpression(n.lambda((o=e[2][0])!=null?u(o,"id"):o,l))+`.html#summary" class="summary">Summary</a>
          </li>
`},28:function(n,l,a,c,s,e){var o,u=n.lambda,r=n.escapeExpression,i=n.lookupProperty||function(t,f){if(Object.prototype.hasOwnProperty.call(t,f))return t[f]};return`          <li class="docs">
            <a href="`+r(u((o=e[2][0])!=null?i(o,"id"):o,l))+".html#"+r(u((o=e[0][0])!=null?i(o,"key"):o,l))+`" class="expand">
              `+r(u((o=e[0][0])!=null?i(o,"name"):o,l))+`
              <span class="icon-expand"></span>
            </a>
            <ul class="`+r(u((o=e[0][0])!=null?i(o,"key"):o,l))+`-list deflist">
`+((o=i(a,"each").call(l??(n.nullContext||{}),(o=e[0][0])!=null?i(o,"nodes"):o,{name:"each",hash:{},fn:n.program(29,s,0,e),inverse:n.noop,data:s,blockParams:e,loc:{start:{line:57,column:14},end:{line:61,column:23}}}))!=null?o:"")+`            </ul>
          </li>
`},29:function(n,l,a,c,s,e){var o,u,r=n.escapeExpression,i=l??(n.nullContext||{}),t=n.hooks.helperMissing,f="function",p=n.lookupProperty||function(m,v){if(Object.prototype.hasOwnProperty.call(m,v))return m[v]};return`                <li>
                  <a href="`+r(n.lambda((o=e[3][0])!=null?p(o,"id"):o,l))+".html#"+r((u=(u=p(a,"anchor")||(l!=null?p(l,"anchor"):l))!=null?u:t,typeof u===f?u.call(i,{name:"anchor",hash:{},data:s,blockParams:e,loc:{start:{line:59,column:44},end:{line:59,column:54}}}):u))+'" title="'+r((u=(u=p(a,"title")||(l!=null?p(l,"title"):l))!=null?u:t,typeof u===f?u.call(i,{name:"title",hash:{},data:s,blockParams:e,loc:{start:{line:59,column:63},end:{line:59,column:72}}}):u))+'" translate="no">'+r((u=(u=p(a,"id")||(l!=null?p(l,"id"):l))!=null?u:t,typeof u===f?u.call(i,{name:"id",hash:{},data:s,blockParams:e,loc:{start:{line:59,column:89},end:{line:59,column:95}}}):u))+`</a>
                </li>
`},compiler:[8,">= 4.3.0"],main:function(n,l,a,c,s,e,o){var u,r=n.lookupProperty||function(i,t){if(Object.prototype.hasOwnProperty.call(i,t))return i[t]};return(u=r(a,"each").call(l??(n.nullContext||{}),l!=null?r(l,"nodes"):l,{name:"each",hash:{},fn:n.program(1,s,2,e,o),inverse:n.noop,data:s,blockParams:e,loc:{start:{line:1,column:0},end:{line:68,column:9}}}))!=null?u:""},useData:!0,useDepths:!0,useBlockParams:!0}),y["tooltip-body"]=d({1:function(n,l,a,c,s){var e,o=n.lookupProperty||function(u,r){if(Object.prototype.hasOwnProperty.call(u,r))return u[r]};return`  <section class="docstring docstring-plain">
    `+n.escapeExpression(n.lambda((e=l!=null?o(l,"hint"):l)!=null?o(e,"description"):e,l))+`
  </section>
`},3:function(n,l,a,c,s){var e,o=n.lambda,u=n.escapeExpression,r=n.lookupProperty||function(i,t){if(Object.prototype.hasOwnProperty.call(i,t))return i[t]};return`  <div class="detail-header">
    <h1 class="signature">
      <span translate="no">`+u(o((e=l!=null?r(l,"hint"):l)!=null?r(e,"title"):e,l))+`</span>
      <div class="version-info" translate="no">`+u(o((e=l!=null?r(l,"hint"):l)!=null?r(e,"version"):e,l))+`</div>
    </h1>
  </div>
`+((e=r(a,"if").call(l??(n.nullContext||{}),(e=l!=null?r(l,"hint"):l)!=null?r(e,"description"):e,{name:"if",hash:{},fn:n.program(4,s,0),inverse:n.noop,data:s,loc:{start:{line:12,column:2},end:{line:16,column:9}}}))!=null?e:"")},4:function(n,l,a,c,s){var e,o=n.lookupProperty||function(u,r){if(Object.prototype.hasOwnProperty.call(u,r))return u[r]};return`    <section class="docstring">
      `+n.escapeExpression(n.lambda((e=l!=null?o(l,"hint"):l)!=null?o(e,"description"):e,l))+`
    </section>
`},compiler:[8,">= 4.3.0"],main:function(n,l,a,c,s){var e,o=n.lookupProperty||function(u,r){if(Object.prototype.hasOwnProperty.call(u,r))return u[r]};return(e=o(a,"if").call(l??(n.nullContext||{}),l!=null?o(l,"isPlain"):l,{name:"if",hash:{},fn:n.program(1,s,0),inverse:n.program(3,s,0),data:s,loc:{start:{line:1,column:0},end:{line:17,column:7}}}))!=null?e:""},useData:!0}),y["tooltip-layout"]=d({compiler:[8,">= 4.3.0"],main:function(n,l,a,c,s){return`<div id="tooltip">
  <div class="tooltip-body"></div>
</div>
`},useData:!0}),y["versions-dropdown"]=d({1:function(n,l,a,c,s){var e,o,u=l??(n.nullContext||{}),r=n.hooks.helperMissing,i="function",t=n.escapeExpression,f=n.lookupProperty||function(p,m){if(Object.prototype.hasOwnProperty.call(p,m))return p[m]};return'      <option translate="no" value="'+t((o=(o=f(a,"url")||(l!=null?f(l,"url"):l))!=null?o:r,typeof o===i?o.call(u,{name:"url",hash:{},data:s,loc:{start:{line:4,column:36},end:{line:4,column:43}}}):o))+'"'+((e=f(a,"if").call(u,l!=null?f(l,"isCurrentVersion"):l,{name:"if",hash:{},fn:n.program(2,s,0),inverse:n.noop,data:s,loc:{start:{line:4,column:44},end:{line:4,column:93}}}))!=null?e:"")+`>
        `+t((o=(o=f(a,"version")||(l!=null?f(l,"version"):l))!=null?o:r,typeof o===i?o.call(u,{name:"version",hash:{},data:s,loc:{start:{line:5,column:8},end:{line:5,column:19}}}):o))+`
      </option>
`},2:function(n,l,a,c,s){return" selected disabled"},compiler:[8,">= 4.3.0"],main:function(n,l,a,c,s){var e,o=n.lookupProperty||function(u,r){if(Object.prototype.hasOwnProperty.call(u,r))return u[r]};return`<form autocomplete="off">
  <select class="sidebar-projectVersionsDropdown">
`+((e=o(a,"each").call(l??(n.nullContext||{}),l!=null?o(l,"nodes"):l,{name:"each",hash:{},fn:n.program(1,s,0),inverse:n.noop,data:s,loc:{start:{line:3,column:4},end:{line:7,column:13}}}))!=null?e:"")+`  </select>
</form>
`},useData:!0})})();})();
