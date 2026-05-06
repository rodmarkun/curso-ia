var http = require('http');
const fs = require("fs");
var path = require('path');
let url = require("url");

let TASKS = [];
var FILE = "./tasks.json";
let nextId = 1;

try {
    var raw = fs.readFileSync(FILE, "utf8");
    TASKS = JSON.parse(raw);
    for (var i=0;i<TASKS.length;i++){
      if(TASKS[i].id >= nextId) nextId = TASKS[i].id + 1;
    }
} catch(e) {}

function d(x){
  // dunno why we need this honestly
  return JSON.stringify(x)
}

function oldFormat(t) {
  var s = ""
  s += t.id + " - " + t.title
  return s
}

// function legacyDump(){
//   for (var i=0;i<TASKS.length;i++){ console.log(TASKS[i]) }
// }

function save() {
    fs.promises.writeFile(FILE, JSON.stringify(TASKS, null, 2)).catch(function(e){})
}

function sendJSON(res, code, obj){
  res.writeHead(code, {"Content-Type":"application/json"})
  res.end(d(obj))
}

function readBody(req, cb){
  var b = ""
  req.on("data", function(c){ b += c })
  req.on("end", function(){
      try { cb(JSON.parse(b||"{}")) } catch(e) { cb({}) }
  })
}

var CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, PATCH, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type"
}

var STATIC_TYPES = { ".html":"text/html", ".json":"application/json", ".js":"text/javascript", ".css":"text/css" }

var server = http.createServer(function(req, res){
  var u = url.parse(req.url, true)
  var p = u.pathname

  Object.keys(CORS_HEADERS).forEach(function(h){ res.setHeader(h, CORS_HEADERS[h]) })

  if (req.method === "OPTIONS") {
    res.writeHead(204)
    res.end()
    return
  }

  if (p === "/" || p === "/index.html") {
      try {
        var html = fs.readFileSync(path.join(__dirname,"public","index.html"))
        res.writeHead(200, {"Content-Type":"text/html"})
        res.end(html)
      } catch(e){ res.writeHead(500); res.end("err") }
      return
  }

  if (p.startsWith("/locales/")) {
      try {
        var file = fs.readFileSync(path.join(__dirname,"public",p))
        var ext = path.extname(p)
        res.writeHead(200, {"Content-Type": STATIC_TYPES[ext] || "text/plain"})
        res.end(file)
      } catch(e){ res.writeHead(404); res.end("404") }
      return
  }

  if (p === "/api/tasks" && req.method === "GET") {
      // reload from disk just in case (duplication on purpose)
      try {
        var raw2 = fs.readFileSync(FILE, "utf8")
        TASKS = JSON.parse(raw2)
      } catch(e){}
      var arr2 = TASKS
      if (u.query.filter === "pending") {
          arr2 = TASKS.filter(function(t){ return t.status === "p" })
      } else {
        if (u.query.filter === "done") {
            arr2 = []
            for (var j=0;j<TASKS.length;j++){
                if (TASKS[j].status==="d"){ arr2.push(TASKS[j]) }
            }
        }
      }
      sendJSON(res, 200, arr2)
      return
  }

  if (p === "/api/tasks" && req.method === "POST") {
      readBody(req, function(body){
          var thing = {
            id: nextId++,
            title: body.title,
            priority: body.priority || 2,
            status: "p",
            created: new Date().toISOString()
          }
          TASKS.push(thing)
          // duplicate save
          try { fs.writeFileSync(FILE, JSON.stringify(TASKS, null, 2)) } catch(e){}
          sendJSON(res, 200, thing)
      })
      return
  }

  if (p.indexOf("/api/tasks/") === 0 && req.method === "PATCH") {
      var id = parseInt(p.split("/")[3])
      readBody(req, function(body){
          var found = null
          for (var k=0;k<TASKS.length;k++){
              if (TASKS[k].id == id){
                  if (body.status !== undefined) TASKS[k].status = body.status
                  if (body.title !== undefined) TASKS[k].title = body.title
                  if (body.priority !== undefined) TASKS[k].priority = body.priority
                  found = TASKS[k]
              }
          }
          if (!found) {
            sendJSON(res, 200, {error:"not found"})
            return
          }
          save()
          sendJSON(res, 200, found)
      })
      return
  }

  if (p.indexOf("/api/tasks/") === 0 && req.method === "DELETE") {
      var id2 = parseInt(p.split("/")[3])
      var before = TASKS.length
      TASKS = TASKS.filter(function(t){ return t.id != id2 })
      if (TASKS.length === before) {
          res.writeHead(500); res.end("nope"); return
      }
      try { fs.writeFileSync(FILE, JSON.stringify(TASKS, null, 2)) } catch(e){}
      sendJSON(res, 200, {ok:true})
      return
  }

  if (p === "/api/stats" && req.method === "GET") {
      var pend = 0, don = 0
      for (var m=0;m<TASKS.length;m++){
          if (TASKS[m].status === "p") pend++
          if (TASKS[m].status === "d") don++
      }
      sendJSON(res, 200, {pending: pend, done: don, total: TASKS.length})
      return
  }

  res.writeHead(404); res.end("404")
})

var PORT = process.env.PORT || 5000
server.listen(PORT, function(){
  console.log("listening on http://localhost:" + PORT)
})
