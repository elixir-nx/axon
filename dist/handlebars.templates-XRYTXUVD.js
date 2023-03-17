(()=>{(function(){var y=Handlebars.template,d=Handlebars.templates=Handlebars.templates||{};d["autocomplete-suggestions"]=y({1:function(n,l,a,c,u){var e,o,r=l??(n.nullContext||{}),s=n.hooks.helperMissing,i="function",t=n.escapeExpression,f=n.lookupProperty||function(p,m){if(Object.prototype.hasOwnProperty.call(p,m))return p[m]};return'    <a href="'+t((o=(o=f(a,"link")||(l!=null?f(l,"link"):l))!=null?o:s,typeof o===i?o.call(r,{name:"link",hash:{},data:u,loc:{start:{line:7,column:13},end:{line:7,column:21}}}):o))+'" class="autocomplete-suggestion" data-index="'+t((o=(o=f(a,"index")||u&&f(u,"index"))!=null?o:s,typeof o===i?o.call(r,{name:"index",hash:{},data:u,loc:{start:{line:7,column:67},end:{line:7,column:77}}}):o))+`" tabindex="-1">
      <div class="title">
        <span translate="no">`+((e=(o=(o=f(a,"title")||(l!=null?f(l,"title"):l))!=null?o:s,typeof o===i?o.call(r,{name:"title",hash:{},data:u,loc:{start:{line:9,column:29},end:{line:9,column:40}}}):o))!=null?e:"")+`</span>
`+((e=f(a,"if").call(r,l!=null?f(l,"label"):l,{name:"if",hash:{},fn:n.program(2,u,0),inverse:n.noop,data:u,loc:{start:{line:10,column:8},end:{line:12,column:15}}}))!=null?e:"")+`      </div>

`+((e=f(a,"if").call(r,l!=null?f(l,"description"):l,{name:"if",hash:{},fn:n.program(4,u,0),inverse:n.noop,data:u,loc:{start:{line:15,column:6},end:{line:19,column:13}}}))!=null?e:"")+`    </a>
`},2:function(n,l,a,c,u){var e,o=n.lookupProperty||function(r,s){if(Object.prototype.hasOwnProperty.call(r,s))return r[s]};return'          <span class="label">('+n.escapeExpression((e=(e=o(a,"label")||(l!=null?o(l,"label"):l))!=null?e:n.hooks.helperMissing,typeof e=="function"?e.call(l??(n.nullContext||{}),{name:"label",hash:{},data:u,loc:{start:{line:11,column:31},end:{line:11,column:40}}}):e))+`)</span>
`},4:function(n,l,a,c,u){var e,o,r=n.lookupProperty||function(s,i){if(Object.prototype.hasOwnProperty.call(s,i))return s[i]};return`        <div class="description" translate="no">
          `+((e=(o=(o=r(a,"description")||(l!=null?r(l,"description"):l))!=null?o:n.hooks.helperMissing,typeof o=="function"?o.call(l??(n.nullContext||{}),{name:"description",hash:{},data:u,loc:{start:{line:17,column:10},end:{line:17,column:27}}}):o))!=null?e:"")+`
        </div>
`},compiler:[8,">= 4.3.0"],main:function(n,l,a,c,u){var e,o,r=l??(n.nullContext||{}),s=n.hooks.helperMissing,i="function",t=n.escapeExpression,f=n.lookupProperty||function(p,m){if(Object.prototype.hasOwnProperty.call(p,m))return p[m]};return`<div class="autocomplete-suggestions">
  <a class="autocomplete-suggestion" href="search.html?q=`+t((o=(o=f(a,"term")||(l!=null?f(l,"term"):l))!=null?o:s,typeof o===i?o.call(r,{name:"term",hash:{},data:u,loc:{start:{line:2,column:57},end:{line:2,column:65}}}):o))+`" data-index="-1" tabindex="-1">
    <div class="title">"<em>`+t((o=(o=f(a,"term")||(l!=null?f(l,"term"):l))!=null?o:s,typeof o===i?o.call(r,{name:"term",hash:{},data:u,loc:{start:{line:3,column:28},end:{line:3,column:36}}}):o))+`</em>"</div>
    <div class="description">Search the documentation</div>
  </a>
`+((e=f(a,"each").call(r,l!=null?f(l,"suggestions"):l,{name:"each",hash:{},fn:n.program(1,u,0),inverse:n.noop,data:u,loc:{start:{line:6,column:2},end:{line:21,column:11}}}))!=null?e:"")+`</div>
`},useData:!0}),d["modal-layout"]=y({compiler:[8,">= 4.3.0"],main:function(n,l,a,c,u){return`<div id="modal" class="modal" tabindex="-1">
  <div class="modal-contents">
    <div class="modal-header">
      <div class="modal-title"></div>
      <button class="modal-close">\xD7</button>
    </div>
    <div class="modal-body">
    </div>
  </div>
</div>
`},useData:!0}),d["quick-switch-modal-body"]=y({compiler:[8,">= 4.3.0"],main:function(n,l,a,c,u){return`<div id="quick-switch-modal-body">
  <i class="ri-search-2-line" aria-hidden="true"></i>
  <input type="text" id="quick-switch-input" class="search-input" placeholder="Jump to..." autocomplete="off" spellcheck="false">
  <div id="quick-switch-results"></div>
</div>
`},useData:!0}),d["quick-switch-results"]=y({1:function(n,l,a,c,u){var e,o=l??(n.nullContext||{}),r=n.hooks.helperMissing,s="function",i=n.escapeExpression,t=n.lookupProperty||function(f,p){if(Object.prototype.hasOwnProperty.call(f,p))return f[p]};return'  <div class="quick-switch-result" data-index="'+i((e=(e=t(a,"index")||u&&t(u,"index"))!=null?e:r,typeof e===s?e.call(o,{name:"index",hash:{},data:u,loc:{start:{line:2,column:47},end:{line:2,column:57}}}):e))+`">
    `+i((e=(e=t(a,"name")||(l!=null?t(l,"name"):l))!=null?e:r,typeof e===s?e.call(o,{name:"name",hash:{},data:u,loc:{start:{line:3,column:4},end:{line:3,column:12}}}):e))+`
  </div>
`},compiler:[8,">= 4.3.0"],main:function(n,l,a,c,u){var e,o=n.lookupProperty||function(r,s){if(Object.prototype.hasOwnProperty.call(r,s))return r[s]};return(e=o(a,"each").call(l??(n.nullContext||{}),l!=null?o(l,"results"):l,{name:"each",hash:{},fn:n.program(1,u,0),inverse:n.noop,data:u,loc:{start:{line:1,column:0},end:{line:5,column:9}}}))!=null?e:""},useData:!0}),d["search-results"]=y({1:function(n,l,a,c,u){var e,o=n.lookupProperty||function(r,s){if(Object.prototype.hasOwnProperty.call(r,s))return r[s]};return"    Search results for <em>"+n.escapeExpression((e=(e=o(a,"value")||(l!=null?o(l,"value"):l))!=null?e:n.hooks.helperMissing,typeof e=="function"?e.call(l??(n.nullContext||{}),{name:"value",hash:{},data:u,loc:{start:{line:3,column:27},end:{line:3,column:36}}}):e))+`</em>
`},3:function(n,l,a,c,u){return`    Invalid search
`},5:function(n,l,a,c,u){var e,o=n.lookupProperty||function(r,s){if(Object.prototype.hasOwnProperty.call(r,s))return r[s]};return(e=o(a,"each").call(l??(n.nullContext||{}),l!=null?o(l,"results"):l,{name:"each",hash:{},fn:n.program(6,u,0),inverse:n.noop,data:u,loc:{start:{line:15,column:2},end:{line:26,column:11}}}))!=null?e:""},6:function(n,l,a,c,u){var e,o=n.lambda,r=n.escapeExpression,s=n.lookupProperty||function(i,t){if(Object.prototype.hasOwnProperty.call(i,t))return i[t]};return`    <div class="result">
      <h2 class="result-id">
        <a href="`+r(o(l!=null?s(l,"ref"):l,l))+`">
          <span translate="no">`+r(o(l!=null?s(l,"title"):l,l))+"</span> <small>("+r(o(l!=null?s(l,"type"):l,l))+`)</small>
        </a>
      </h2>
`+((e=s(a,"each").call(l??(n.nullContext||{}),l!=null?s(l,"excerpts"):l,{name:"each",hash:{},fn:n.program(7,u,0),inverse:n.noop,data:u,loc:{start:{line:22,column:8},end:{line:24,column:17}}}))!=null?e:"")+`    </div>
`},7:function(n,l,a,c,u){var e;return'          <p class="result-elem">'+((e=n.lambda(l,l))!=null?e:"")+`</p>
`},9:function(n,l,a,c,u){var e,o=n.lookupProperty||function(r,s){if(Object.prototype.hasOwnProperty.call(r,s))return r[s]};return((e=(o(a,"isArray")||l&&o(l,"isArray")||n.hooks.helperMissing).call(l??(n.nullContext||{}),l!=null?o(l,"results"):l,{name:"isArray",hash:{},fn:n.program(10,u,0),inverse:n.program(12,u,0),data:u,loc:{start:{line:28,column:2},end:{line:34,column:14}}}))!=null?e:"")+`
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
`},10:function(n,l,a,c,u){var e,o=n.lookupProperty||function(r,s){if(Object.prototype.hasOwnProperty.call(r,s))return r[s]};return"    <p>Sorry, we couldn't find anything for <em>"+n.escapeExpression((e=(e=o(a,"value")||(l!=null?o(l,"value"):l))!=null?e:n.hooks.helperMissing,typeof e=="function"?e.call(l??(n.nullContext||{}),{name:"value",hash:{},data:u,loc:{start:{line:29,column:48},end:{line:29,column:57}}}):e))+`</em>.</p>
`},12:function(n,l,a,c,u){var e,o=n.lookupProperty||function(r,s){if(Object.prototype.hasOwnProperty.call(r,s))return r[s]};return(e=o(a,"if").call(l??(n.nullContext||{}),l!=null?o(l,"value"):l,{name:"if",hash:{},fn:n.program(13,u,0),inverse:n.program(15,u,0),data:u,loc:{start:{line:30,column:2},end:{line:34,column:2}}}))!=null?e:""},13:function(n,l,a,c,u){var e,o=n.lookupProperty||function(r,s){if(Object.prototype.hasOwnProperty.call(r,s))return r[s]};return"    <p>Invalid search: "+n.escapeExpression((e=(e=o(a,"errorMessage")||(l!=null?o(l,"errorMessage"):l))!=null?e:n.hooks.helperMissing,typeof e=="function"?e.call(l??(n.nullContext||{}),{name:"errorMessage",hash:{},data:u,loc:{start:{line:31,column:23},end:{line:31,column:39}}}):e))+`.</p>
`},15:function(n,l,a,c,u){return`    <p>Please type something into the search bar to perform a search.</p>
  `},compiler:[8,">= 4.3.0"],main:function(n,l,a,c,u){var e,o=l??(n.nullContext||{}),r=n.lookupProperty||function(s,i){if(Object.prototype.hasOwnProperty.call(s,i))return s[i]};return`<h1>
`+((e=r(a,"if").call(o,l!=null?r(l,"value"):l,{name:"if",hash:{},fn:n.program(1,u,0),inverse:n.program(3,u,0),data:u,loc:{start:{line:2,column:2},end:{line:6,column:9}}}))!=null?e:"")+`
  <button class="icon-action display-settings">
    <i class="ri-settings-3-line"></i>
    <span class="sr-only">Settings</span>
  </button>
</h1>

`+((e=(r(a,"isNonEmptyArray")||l&&r(l,"isNonEmptyArray")||n.hooks.helperMissing).call(o,l!=null?r(l,"results"):l,{name:"isNonEmptyArray",hash:{},fn:n.program(5,u,0),inverse:n.program(9,u,0),data:u,loc:{start:{line:14,column:0},end:{line:49,column:20}}}))!=null?e:"")},useData:!0}),d["settings-modal-body"]=y({1:function(n,l,a,c,u){var e,o=n.lookupProperty||function(r,s){if(Object.prototype.hasOwnProperty.call(r,s))return r[s]};return(e=o(a,"if").call(l??(n.nullContext||{}),l!=null?o(l,"description"):l,{name:"if",hash:{},fn:n.program(2,u,0),inverse:n.noop,data:u,loc:{start:{line:40,column:6},end:{line:53,column:13}}}))!=null?e:""},2:function(n,l,a,c,u){var e,o=n.lookupProperty||function(r,s){if(Object.prototype.hasOwnProperty.call(r,s))return r[s]};return`        <dl class="shortcut-row">
          <dd class="shortcut-description">
            `+n.escapeExpression(n.lambda(l!=null?o(l,"description"):l,l))+`
          </dd>
          <dt class="shortcut-keys">
`+((e=o(a,"if").call(l??(n.nullContext||{}),l!=null?o(l,"displayAs"):l,{name:"if",hash:{},fn:n.program(3,u,0),inverse:n.program(5,u,0),data:u,loc:{start:{line:46,column:12},end:{line:50,column:19}}}))!=null?e:"")+`          </dt>
        </dl>
`},3:function(n,l,a,c,u){var e,o=n.lookupProperty||function(r,s){if(Object.prototype.hasOwnProperty.call(r,s))return r[s]};return"              "+((e=n.lambda(l!=null?o(l,"displayAs"):l,l))!=null?e:"")+`
`},5:function(n,l,a,c,u){var e=n.lookupProperty||function(o,r){if(Object.prototype.hasOwnProperty.call(o,r))return o[r]};return"              <kbd><kbd>"+n.escapeExpression(n.lambda(l!=null?e(l,"key"):l,l))+`</kbd></kbd>
`},compiler:[8,">= 4.3.0"],main:function(n,l,a,c,u){var e,o=n.lookupProperty||function(r,s){if(Object.prototype.hasOwnProperty.call(r,s))return r[s]};return`<div id="settings-modal-content">
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
`+((e=o(a,"each").call(l??(n.nullContext||{}),l!=null?o(l,"shortcuts"):l,{name:"each",hash:{},fn:n.program(1,u,0),inverse:n.noop,data:u,loc:{start:{line:39,column:4},end:{line:54,column:13}}}))!=null?e:"")+`  </div>
</div>
`},useData:!0}),d["sidebar-items"]=y({1:function(n,l,a,c,u,e,o){var r,s=l??(n.nullContext||{}),i=n.hooks.helperMissing,t=n.lookupProperty||function(f,p){if(Object.prototype.hasOwnProperty.call(f,p))return f[p]};return((r=(t(a,"groupChanged")||l&&t(l,"groupChanged")||i).call(s,o[1],(r=e[0][0])!=null?t(r,"group"):r,{name:"groupChanged",hash:{},fn:n.program(2,u,0,e,o),inverse:n.noop,data:u,blockParams:e,loc:{start:{line:2,column:2},end:{line:6,column:19}}}))!=null?r:"")+`
`+((r=(t(a,"nestingChanged")||l&&t(l,"nestingChanged")||i).call(s,o[1],e[0][0],{name:"nestingChanged",hash:{},fn:n.program(7,u,0,e,o),inverse:n.noop,data:u,blockParams:e,loc:{start:{line:8,column:2},end:{line:10,column:21}}}))!=null?r:"")+`
  <li class="`+((r=(t(a,"isLocal")||l&&t(l,"isLocal")||i).call(s,(r=e[0][0])!=null?t(r,"id"):r,{name:"isLocal",hash:{},fn:n.program(9,u,0,e,o),inverse:n.noop,data:u,blockParams:e,loc:{start:{line:12,column:13},end:{line:12,column:62}}}))!=null?r:"")+`">
    <a href="`+n.escapeExpression(n.lambda((r=e[0][0])!=null?t(r,"id"):r,l))+".html"+((r=(t(a,"isLocal")||l&&t(l,"isLocal")||i).call(s,(r=e[0][0])!=null?t(r,"id"):r,{name:"isLocal",hash:{},fn:n.program(11,u,0,e,o),inverse:n.noop,data:u,blockParams:e,loc:{start:{line:13,column:29},end:{line:13,column:69}}}))!=null?r:"")+'" class="expand" '+((r=(t(a,"isArray")||l&&t(l,"isArray")||i).call(s,(r=e[0][0])!=null?t(r,"headers"):r,{name:"isArray",hash:{},fn:n.program(3,u,0,e,o),inverse:n.program(5,u,0,e,o),data:u,blockParams:e,loc:{start:{line:13,column:86},end:{line:13,column:145}}}))!=null?r:"")+`>
`+((r=t(a,"if").call(s,(r=e[0][0])!=null?t(r,"nested_title"):r,{name:"if",hash:{},fn:n.program(13,u,0,e,o),inverse:n.program(15,u,0,e,o),data:u,blockParams:e,loc:{start:{line:14,column:6},end:{line:18,column:13}}}))!=null?r:"")+((r=(t(a,"isEmptyArray")||l&&t(l,"isEmptyArray")||i).call(s,(r=e[0][0])!=null?t(r,"headers"):r,{name:"isEmptyArray",hash:{},fn:n.program(3,u,0,e,o),inverse:n.program(17,u,0,e,o),data:u,blockParams:e,loc:{start:{line:19,column:6},end:{line:22,column:23}}}))!=null?r:"")+`    </a>

`+((r=(t(a,"isArray")||l&&t(l,"isArray")||i).call(s,(r=e[0][0])!=null?t(r,"headers"):r,{name:"isArray",hash:{},fn:n.program(19,u,0,e,o),inverse:n.program(23,u,0,e,o),data:u,blockParams:e,loc:{start:{line:25,column:4},end:{line:73,column:16}}}))!=null?r:"")+`  </li>
`},2:function(n,l,a,c,u,e){var o,r=n.lookupProperty||function(s,i){if(Object.prototype.hasOwnProperty.call(s,i))return s[i]};return'    <li class="group" '+((o=(r(a,"isArray")||l&&r(l,"isArray")||n.hooks.helperMissing).call(l??(n.nullContext||{}),(o=e[1][0])!=null?r(o,"headers"):o,{name:"isArray",hash:{},fn:n.program(3,u,0,e),inverse:n.program(5,u,0,e),data:u,blockParams:e,loc:{start:{line:3,column:22},end:{line:3,column:81}}}))!=null?o:"")+`>
      `+n.escapeExpression(n.lambda((o=e[1][0])!=null?r(o,"group"):o,l))+`
    </li>
`},3:function(n,l,a,c,u){return""},5:function(n,l,a,c,u){return'translate="no"'},7:function(n,l,a,c,u,e){var o,r=n.lookupProperty||function(s,i){if(Object.prototype.hasOwnProperty.call(s,i))return s[i]};return'    <li class="nesting-context" aria-hidden="true" translate="no">'+n.escapeExpression(n.lambda((o=e[1][0])!=null?r(o,"nested_context"):o,l))+`</li>
`},9:function(n,l,a,c,u){return"current-page open"},11:function(n,l,a,c,u){return"#content"},13:function(n,l,a,c,u,e){var o,r=n.lookupProperty||function(s,i){if(Object.prototype.hasOwnProperty.call(s,i))return s[i]};return"        "+((o=n.lambda((o=e[1][0])!=null?r(o,"nested_title"):o,l))!=null?o:"")+`
`},15:function(n,l,a,c,u,e){var o,r=n.lookupProperty||function(s,i){if(Object.prototype.hasOwnProperty.call(s,i))return s[i]};return"        "+((o=n.lambda((o=e[1][0])!=null?r(o,"title"):o,l))!=null?o:"")+`
`},17:function(n,l,a,c,u){return`        <span class="icon-expand"></span>
`},19:function(n,l,a,c,u,e){var o,r=n.lookupProperty||function(s,i){if(Object.prototype.hasOwnProperty.call(s,i))return s[i]};return(o=(r(a,"isNonEmptyArray")||l&&r(l,"isNonEmptyArray")||n.hooks.helperMissing).call(l??(n.nullContext||{}),(o=e[1][0])!=null?r(o,"headers"):o,{name:"isNonEmptyArray",hash:{},fn:n.program(20,u,0,e),inverse:n.noop,data:u,blockParams:e,loc:{start:{line:26,column:6},end:{line:34,column:26}}}))!=null?o:""},20:function(n,l,a,c,u,e){var o,r=n.lookupProperty||function(s,i){if(Object.prototype.hasOwnProperty.call(s,i))return s[i]};return`        <ul>
`+((o=r(a,"each").call(l??(n.nullContext||{}),(o=e[2][0])!=null?r(o,"headers"):o,{name:"each",hash:{},fn:n.program(21,u,0,e),inverse:n.noop,data:u,blockParams:e,loc:{start:{line:28,column:10},end:{line:32,column:19}}}))!=null?o:"")+`        </ul>
`},21:function(n,l,a,c,u,e){var o,r,s=l??(n.nullContext||{}),i=n.hooks.helperMissing,t="function",f=n.lookupProperty||function(p,m){if(Object.prototype.hasOwnProperty.call(p,m))return p[m]};return`            <li>
              <a href="`+n.escapeExpression(n.lambda((o=e[3][0])!=null?f(o,"id"):o,l))+".html#"+((o=(r=(r=f(a,"anchor")||(l!=null?f(l,"anchor"):l))!=null?r:i,typeof r===t?r.call(s,{name:"anchor",hash:{},data:u,blockParams:e,loc:{start:{line:30,column:40},end:{line:30,column:52}}}):r))!=null?o:"")+'">'+((o=(r=(r=f(a,"id")||(l!=null?f(l,"id"):l))!=null?r:i,typeof r===t?r.call(s,{name:"id",hash:{},data:u,blockParams:e,loc:{start:{line:30,column:54},end:{line:30,column:62}}}):r))!=null?o:"")+`</a>
            </li>
`},23:function(n,l,a,c,u,e){var o,r=l??(n.nullContext||{}),s=n.hooks.helperMissing,i=n.lookupProperty||function(t,f){if(Object.prototype.hasOwnProperty.call(t,f))return t[f]};return`      <ul>
`+((o=(i(a,"showSections")||l&&i(l,"showSections")||s).call(r,e[1][0],{name:"showSections",hash:{},fn:n.program(24,u,0,e),inverse:n.noop,data:u,blockParams:e,loc:{start:{line:37,column:8},end:{line:51,column:25}}}))!=null?o:"")+((o=(i(a,"showSummary")||l&&i(l,"showSummary")||s).call(r,e[1][0],{name:"showSummary",hash:{},fn:n.program(29,u,0,e),inverse:n.noop,data:u,blockParams:e,loc:{start:{line:52,column:8},end:{line:56,column:24}}}))!=null?o:"")+((o=i(a,"each").call(r,(o=e[1][0])!=null?i(o,"nodeGroups"):o,{name:"each",hash:{},fn:n.program(31,u,1,e),inverse:n.noop,data:u,blockParams:e,loc:{start:{line:57,column:8},end:{line:71,column:17}}}))!=null?o:"")+`      </ul>
`},24:function(n,l,a,c,u,e){var o,r=l??(n.nullContext||{}),s=n.lookupProperty||function(i,t){if(Object.prototype.hasOwnProperty.call(i,t))return i[t]};return'          <li class="docs '+((o=(s(a,"isLocal")||l&&s(l,"isLocal")||n.hooks.helperMissing).call(r,(o=e[2][0])!=null?s(o,"id"):o,{name:"isLocal",hash:{},fn:n.program(25,u,0,e),inverse:n.noop,data:u,blockParams:e,loc:{start:{line:38,column:26},end:{line:38,column:62}}}))!=null?o:"")+`">
            <a href="`+n.escapeExpression(n.lambda((o=e[2][0])!=null?s(o,"id"):o,l))+`.html#content" class="expand">
              Sections
              <span class="icon-expand"></span>
            </a>
            <ul class="sections-list deflist">
`+((o=s(a,"each").call(r,l!=null?s(l,"sections"):l,{name:"each",hash:{},fn:n.program(27,u,0,e),inverse:n.noop,data:u,blockParams:e,loc:{start:{line:44,column:14},end:{line:48,column:23}}}))!=null?o:"")+`            </ul>
          </li>
`},25:function(n,l,a,c,u){return"open"},27:function(n,l,a,c,u,e){var o,r,s=n.escapeExpression,i=l??(n.nullContext||{}),t=n.hooks.helperMissing,f="function",p=n.lookupProperty||function(m,v){if(Object.prototype.hasOwnProperty.call(m,v))return m[v]};return`                <li>
                  <a href="`+s(n.lambda((o=e[3][0])!=null?p(o,"id"):o,l))+".html#"+s((r=(r=p(a,"anchor")||(l!=null?p(l,"anchor"):l))!=null?r:t,typeof r===f?r.call(i,{name:"anchor",hash:{},data:u,blockParams:e,loc:{start:{line:46,column:44},end:{line:46,column:54}}}):r))+'">'+((o=(r=(r=p(a,"id")||(l!=null?p(l,"id"):l))!=null?r:t,typeof r===f?r.call(i,{name:"id",hash:{},data:u,blockParams:e,loc:{start:{line:46,column:56},end:{line:46,column:64}}}):r))!=null?o:"")+`</a>
                </li>
`},29:function(n,l,a,c,u,e){var o,r=n.lookupProperty||function(s,i){if(Object.prototype.hasOwnProperty.call(s,i))return s[i]};return`          <li>
            <a href="`+n.escapeExpression(n.lambda((o=e[2][0])!=null?r(o,"id"):o,l))+`.html#summary" class="summary">Summary</a>
          </li>
`},31:function(n,l,a,c,u,e){var o,r=n.lambda,s=n.escapeExpression,i=n.lookupProperty||function(t,f){if(Object.prototype.hasOwnProperty.call(t,f))return t[f]};return`          <li class="docs">
            <a href="`+s(r((o=e[2][0])!=null?i(o,"id"):o,l))+".html#"+s(r((o=e[0][0])!=null?i(o,"key"):o,l))+`" class="expand">
              `+s(r((o=e[0][0])!=null?i(o,"name"):o,l))+`
              <span class="icon-expand"></span>
            </a>
            <ul class="`+s(r((o=e[0][0])!=null?i(o,"key"):o,l))+`-list deflist">
`+((o=i(a,"each").call(l??(n.nullContext||{}),(o=e[0][0])!=null?i(o,"nodes"):o,{name:"each",hash:{},fn:n.program(32,u,0,e),inverse:n.noop,data:u,blockParams:e,loc:{start:{line:64,column:14},end:{line:68,column:23}}}))!=null?o:"")+`            </ul>
          </li>
`},32:function(n,l,a,c,u,e){var o,r,s=n.escapeExpression,i=l??(n.nullContext||{}),t=n.hooks.helperMissing,f="function",p=n.lookupProperty||function(m,v){if(Object.prototype.hasOwnProperty.call(m,v))return m[v]};return`                <li>
                  <a href="`+s(n.lambda((o=e[3][0])!=null?p(o,"id"):o,l))+".html#"+s((r=(r=p(a,"anchor")||(l!=null?p(l,"anchor"):l))!=null?r:t,typeof r===f?r.call(i,{name:"anchor",hash:{},data:u,blockParams:e,loc:{start:{line:66,column:44},end:{line:66,column:54}}}):r))+'" title="'+s((r=(r=p(a,"title")||(l!=null?p(l,"title"):l))!=null?r:t,typeof r===f?r.call(i,{name:"title",hash:{},data:u,blockParams:e,loc:{start:{line:66,column:63},end:{line:66,column:72}}}):r))+'" translate="no">'+s((r=(r=p(a,"id")||(l!=null?p(l,"id"):l))!=null?r:t,typeof r===f?r.call(i,{name:"id",hash:{},data:u,blockParams:e,loc:{start:{line:66,column:89},end:{line:66,column:95}}}):r))+`</a>
                </li>
`},compiler:[8,">= 4.3.0"],main:function(n,l,a,c,u,e,o){var r,s=n.lookupProperty||function(i,t){if(Object.prototype.hasOwnProperty.call(i,t))return i[t]};return(r=s(a,"each").call(l??(n.nullContext||{}),l!=null?s(l,"nodes"):l,{name:"each",hash:{},fn:n.program(1,u,2,e,o),inverse:n.noop,data:u,blockParams:e,loc:{start:{line:1,column:0},end:{line:75,column:9}}}))!=null?r:""},useData:!0,useDepths:!0,useBlockParams:!0}),d["tooltip-body"]=y({1:function(n,l,a,c,u){var e,o=n.lookupProperty||function(r,s){if(Object.prototype.hasOwnProperty.call(r,s))return r[s]};return`  <section class="docstring docstring-plain">
    `+n.escapeExpression(n.lambda((e=l!=null?o(l,"hint"):l)!=null?o(e,"description"):e,l))+`
  </section>
`},3:function(n,l,a,c,u){var e,o=n.lambda,r=n.escapeExpression,s=n.lookupProperty||function(i,t){if(Object.prototype.hasOwnProperty.call(i,t))return i[t]};return`  <div class="detail-header">
    <h1 class="signature">
      <span translate="no">`+r(o((e=l!=null?s(l,"hint"):l)!=null?s(e,"title"):e,l))+`</span>
      <div class="version-info" translate="no">`+r(o((e=l!=null?s(l,"hint"):l)!=null?s(e,"version"):e,l))+`</div>
    </h1>
  </div>
`+((e=s(a,"if").call(l??(n.nullContext||{}),(e=l!=null?s(l,"hint"):l)!=null?s(e,"description"):e,{name:"if",hash:{},fn:n.program(4,u,0),inverse:n.noop,data:u,loc:{start:{line:12,column:2},end:{line:16,column:9}}}))!=null?e:"")},4:function(n,l,a,c,u){var e,o=n.lookupProperty||function(r,s){if(Object.prototype.hasOwnProperty.call(r,s))return r[s]};return`    <section class="docstring">
      `+n.escapeExpression(n.lambda((e=l!=null?o(l,"hint"):l)!=null?o(e,"description"):e,l))+`
    </section>
`},compiler:[8,">= 4.3.0"],main:function(n,l,a,c,u){var e,o=n.lookupProperty||function(r,s){if(Object.prototype.hasOwnProperty.call(r,s))return r[s]};return(e=o(a,"if").call(l??(n.nullContext||{}),l!=null?o(l,"isPlain"):l,{name:"if",hash:{},fn:n.program(1,u,0),inverse:n.program(3,u,0),data:u,loc:{start:{line:1,column:0},end:{line:17,column:7}}}))!=null?e:""},useData:!0}),d["tooltip-layout"]=y({compiler:[8,">= 4.3.0"],main:function(n,l,a,c,u){return`<div id="tooltip">
  <div class="tooltip-body"></div>
</div>
`},useData:!0}),d["versions-dropdown"]=y({1:function(n,l,a,c,u){var e,o,r=l??(n.nullContext||{}),s=n.hooks.helperMissing,i="function",t=n.escapeExpression,f=n.lookupProperty||function(p,m){if(Object.prototype.hasOwnProperty.call(p,m))return p[m]};return'      <option translate="no" value="'+t((o=(o=f(a,"url")||(l!=null?f(l,"url"):l))!=null?o:s,typeof o===i?o.call(r,{name:"url",hash:{},data:u,loc:{start:{line:4,column:36},end:{line:4,column:43}}}):o))+'"'+((e=f(a,"if").call(r,l!=null?f(l,"isCurrentVersion"):l,{name:"if",hash:{},fn:n.program(2,u,0),inverse:n.noop,data:u,loc:{start:{line:4,column:44},end:{line:4,column:93}}}))!=null?e:"")+`>
        `+t((o=(o=f(a,"version")||(l!=null?f(l,"version"):l))!=null?o:s,typeof o===i?o.call(r,{name:"version",hash:{},data:u,loc:{start:{line:5,column:8},end:{line:5,column:19}}}):o))+`
      </option>
`},2:function(n,l,a,c,u){return" selected disabled"},compiler:[8,">= 4.3.0"],main:function(n,l,a,c,u){var e,o=n.lookupProperty||function(r,s){if(Object.prototype.hasOwnProperty.call(r,s))return r[s]};return`<form autocomplete="off">
  <select class="sidebar-projectVersionsDropdown">
`+((e=o(a,"each").call(l??(n.nullContext||{}),l!=null?o(l,"nodes"):l,{name:"each",hash:{},fn:n.program(1,u,0),inverse:n.noop,data:u,loc:{start:{line:3,column:4},end:{line:7,column:13}}}))!=null?e:"")+`  </select>
</form>
`},useData:!0})})();})();
