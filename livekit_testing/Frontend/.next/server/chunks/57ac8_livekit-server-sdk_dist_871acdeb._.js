module.exports = {

"[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/grants.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "claimsToJwtPayload": (()=>claimsToJwtPayload),
    "trackSourceToString": (()=>trackSourceToString)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$module__evaluation$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+protocol@1.39.2/node_modules/@livekit/protocol/dist/index.mjs [app-route] (ecmascript) <module evaluation>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+protocol@1.39.2/node_modules/@livekit/protocol/dist/index.mjs [app-route] (ecmascript) <locals>");
;
function trackSourceToString(source) {
    switch(source){
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["TrackSource"].CAMERA:
            return "camera";
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["TrackSource"].MICROPHONE:
            return "microphone";
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["TrackSource"].SCREEN_SHARE:
            return "screen_share";
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["TrackSource"].SCREEN_SHARE_AUDIO:
            return "screen_share_audio";
        default:
            throw new TypeError(`Cannot convert TrackSource ${source} to string`);
    }
}
function claimsToJwtPayload(grant) {
    var _a;
    const claim = {
        ...grant
    };
    if (Array.isArray((_a = claim.video) == null ? void 0 : _a.canPublishSources)) {
        claim.video.canPublishSources = claim.video.canPublishSources.map(trackSourceToString);
    }
    return claim;
}
;
 //# sourceMappingURL=grants.js.map
}}),
"[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/AccessToken.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "AccessToken": (()=>AccessToken),
    "TokenVerifier": (()=>TokenVerifier)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$jose$40$5$2e$10$2e$0$2f$node_modules$2f$jose$2f$dist$2f$node$2f$esm$2f$jwt$2f$sign$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/jose@5.10.0/node_modules/jose/dist/node/esm/jwt/sign.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$jose$40$5$2e$10$2e$0$2f$node_modules$2f$jose$2f$dist$2f$node$2f$esm$2f$jwt$2f$verify$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/jose@5.10.0/node_modules/jose/dist/node/esm/jwt/verify.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$grants$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/grants.js [app-route] (ecmascript)");
;
;
const defaultTTL = `6h`;
const defaultClockToleranceSeconds = 10;
class AccessToken {
    /**
   * Creates a new AccessToken
   * @param apiKey - API Key, can be set in env LIVEKIT_API_KEY
   * @param apiSecret - Secret, can be set in env LIVEKIT_API_SECRET
   */ constructor(apiKey, apiSecret, options){
        if (!apiKey) {
            apiKey = process.env.LIVEKIT_API_KEY;
        }
        if (!apiSecret) {
            apiSecret = process.env.LIVEKIT_API_SECRET;
        }
        if (!apiKey || !apiSecret) {
            throw Error("api-key and api-secret must be set");
        } else if (typeof document !== "undefined") {
            console.error("You should not include your API secret in your web client bundle.\n\nYour web client should request a token from your backend server which should then use the API secret to generate a token. See https://docs.livekit.io/client/connect/");
        }
        this.apiKey = apiKey;
        this.apiSecret = apiSecret;
        this.grants = {};
        this.identity = options == null ? void 0 : options.identity;
        this.ttl = (options == null ? void 0 : options.ttl) || defaultTTL;
        if (typeof this.ttl === "number") {
            this.ttl = `${this.ttl}s`;
        }
        if (options == null ? void 0 : options.metadata) {
            this.metadata = options.metadata;
        }
        if (options == null ? void 0 : options.attributes) {
            this.attributes = options.attributes;
        }
        if (options == null ? void 0 : options.name) {
            this.name = options.name;
        }
    }
    /**
   * Adds a video grant to this token.
   * @param grant -
   */ addGrant(grant) {
        this.grants.video = {
            ...this.grants.video ?? {},
            ...grant
        };
    }
    /**
   * Adds a SIP grant to this token.
   * @param grant -
   */ addSIPGrant(grant) {
        this.grants.sip = {
            ...this.grants.sip ?? {},
            ...grant
        };
    }
    get name() {
        return this.grants.name;
    }
    set name(name) {
        this.grants.name = name;
    }
    get metadata() {
        return this.grants.metadata;
    }
    /**
   * Set metadata to be passed to the Participant, used only when joining the room
   */ set metadata(md) {
        this.grants.metadata = md;
    }
    get attributes() {
        return this.grants.attributes;
    }
    set attributes(attrs) {
        this.grants.attributes = attrs;
    }
    get kind() {
        return this.grants.kind;
    }
    set kind(kind) {
        this.grants.kind = kind;
    }
    get sha256() {
        return this.grants.sha256;
    }
    set sha256(sha) {
        this.grants.sha256 = sha;
    }
    get roomPreset() {
        return this.grants.roomPreset;
    }
    set roomPreset(preset) {
        this.grants.roomPreset = preset;
    }
    get roomConfig() {
        return this.grants.roomConfig;
    }
    set roomConfig(config) {
        this.grants.roomConfig = config;
    }
    /**
   * @returns JWT encoded token
   */ async toJwt() {
        var _a;
        const secret = new TextEncoder().encode(this.apiSecret);
        const jwt = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$jose$40$5$2e$10$2e$0$2f$node_modules$2f$jose$2f$dist$2f$node$2f$esm$2f$jwt$2f$sign$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["SignJWT"]((0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$grants$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["claimsToJwtPayload"])(this.grants)).setProtectedHeader({
            alg: "HS256"
        }).setIssuer(this.apiKey).setExpirationTime(this.ttl).setNotBefore(0);
        if (this.identity) {
            jwt.setSubject(this.identity);
        } else if ((_a = this.grants.video) == null ? void 0 : _a.roomJoin) {
            throw Error("identity is required for join but not set");
        }
        return jwt.sign(secret);
    }
}
class TokenVerifier {
    constructor(apiKey, apiSecret){
        this.apiKey = apiKey;
        this.apiSecret = apiSecret;
    }
    async verify(token, clockTolerance = defaultClockToleranceSeconds) {
        const secret = new TextEncoder().encode(this.apiSecret);
        const { payload } = await (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$jose$40$5$2e$10$2e$0$2f$node_modules$2f$jose$2f$dist$2f$node$2f$esm$2f$jwt$2f$verify$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["jwtVerify"])(token, secret, {
            issuer: this.apiKey,
            clockTolerance
        });
        if (!payload) {
            throw Error("invalid token");
        }
        return payload;
    }
}
;
 //# sourceMappingURL=AccessToken.js.map
}}),
"[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/ServiceBase.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "ServiceBase": (()=>ServiceBase)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$AccessToken$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/AccessToken.js [app-route] (ecmascript)");
;
class ServiceBase {
    /**
   * @param apiKey - API Key.
   * @param secret - API Secret.
   * @param ttl - token TTL
   */ constructor(apiKey, secret, ttl){
        this.apiKey = apiKey;
        this.secret = secret;
        this.ttl = ttl || "10m";
    }
    async authHeader(grant, sip) {
        const at = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$AccessToken$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["AccessToken"](this.apiKey, this.secret, {
            ttl: this.ttl
        });
        if (grant) {
            at.addGrant(grant);
        }
        if (sip) {
            at.addSIPGrant(sip);
        }
        return {
            Authorization: `Bearer ${await at.toJwt()}`
        };
    }
}
;
 //# sourceMappingURL=ServiceBase.js.map
}}),
"[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/TwirpRPC.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "TwirpError": (()=>TwirpError),
    "TwirpRpc": (()=>TwirpRpc),
    "livekitPackage": (()=>livekitPackage)
});
const defaultPrefix = "/twirp";
const livekitPackage = "livekit";
class TwirpError extends Error {
    constructor(name, message, status, code, metadata){
        super(message);
        this.name = name;
        this.status = status;
        this.code = code;
        this.metadata = metadata;
    }
}
class TwirpRpc {
    constructor(host, pkg, prefix){
        if (host.startsWith("ws")) {
            host = host.replace("ws", "http");
        }
        this.host = host;
        this.pkg = pkg;
        this.prefix = prefix || defaultPrefix;
    }
    async request(service, method, data, headers, timeout = 60) {
        const path = `${this.prefix}/${this.pkg}.${service}/${method}`;
        const url = new URL(path, this.host);
        const init = {
            method: "POST",
            headers: {
                "Content-Type": "application/json;charset=UTF-8",
                ...headers
            },
            body: JSON.stringify(data)
        };
        if (timeout) {
            init.signal = AbortSignal.timeout(timeout * 1e3);
        }
        const response = await fetch(url, init);
        if (!response.ok) {
            const isJson = response.headers.get("content-type") === "application/json";
            let errorMessage = "Unknown internal error";
            let errorCode = void 0;
            let metadata = void 0;
            try {
                if (isJson) {
                    const parsedError = await response.json();
                    if ("msg" in parsedError) {
                        errorMessage = parsedError.msg;
                    }
                    if ("code" in parsedError) {
                        errorCode = parsedError.code;
                    }
                    if ("meta" in parsedError) {
                        metadata = parsedError.meta;
                    }
                } else {
                    errorMessage = await response.text();
                }
            } catch (e) {
                console.debug(`Error when trying to parse error message, using defaults`, e);
            }
            throw new TwirpError(response.statusText, errorMessage, response.status, errorCode, metadata);
        }
        const parsedResp = await response.json();
        const camelcaseKeys = await __turbopack_context__.r("[project]/node_modules/.pnpm/camelcase-keys@9.1.3/node_modules/camelcase-keys/index.js [app-route] (ecmascript, async loader)")(__turbopack_context__.i).then((mod)=>mod.default);
        return camelcaseKeys(parsedResp, {
            deep: true
        });
    }
}
;
 //# sourceMappingURL=TwirpRPC.js.map
}}),
"[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/AgentDispatchClient.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "AgentDispatchClient": (()=>AgentDispatchClient)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$module__evaluation$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+protocol@1.39.2/node_modules/@livekit/protocol/dist/index.mjs [app-route] (ecmascript) <module evaluation>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+protocol@1.39.2/node_modules/@livekit/protocol/dist/index.mjs [app-route] (ecmascript) <locals>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$ServiceBase$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/ServiceBase.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$TwirpRPC$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/TwirpRPC.js [app-route] (ecmascript)");
;
;
;
const svc = "AgentDispatchService";
class AgentDispatchClient extends __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$ServiceBase$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ServiceBase"] {
    /**
   * @param host - hostname including protocol. i.e. 'https://<project>.livekit.cloud'
   * @param apiKey - API Key, can be set in env var LIVEKIT_API_KEY
   * @param secret - API Secret, can be set in env var LIVEKIT_API_SECRET
   */ constructor(host, apiKey, secret){
        super(apiKey, secret);
        this.rpc = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$TwirpRPC$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["TwirpRpc"](host, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$TwirpRPC$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["livekitPackage"]);
    }
    /**
   * Create an explicit dispatch for an agent to join a room. To use explicit
   * dispatch, your agent must be registered with an `agentName`.
   * @param roomName - name of the room to dispatch to
   * @param agentName - name of the agent to dispatch
   * @param options - optional metadata to send along with the dispatch
   * @returns the dispatch that was created
   */ async createDispatch(roomName, agentName, options) {
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["CreateAgentDispatchRequest"]({
            room: roomName,
            agentName,
            metadata: options == null ? void 0 : options.metadata
        }).toJson();
        const data = await this.rpc.request(svc, "CreateDispatch", req, await this.authHeader({
            roomAdmin: true,
            room: roomName
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["AgentDispatch"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * Delete an explicit dispatch for an agent in a room.
   * @param dispatchId - id of the dispatch to delete
   * @param roomName - name of the room the dispatch is for
   */ async deleteDispatch(dispatchId, roomName) {
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["DeleteAgentDispatchRequest"]({
            dispatchId,
            room: roomName
        }).toJson();
        await this.rpc.request(svc, "DeleteDispatch", req, await this.authHeader({
            roomAdmin: true,
            room: roomName
        }));
    }
    /**
   * Get an Agent dispatch by ID
   * @param dispatchId - id of the dispatch to get
   * @param roomName - name of the room the dispatch is for
   * @returns the dispatch that was found, or undefined if not found
   */ async getDispatch(dispatchId, roomName) {
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListAgentDispatchRequest"]({
            dispatchId,
            room: roomName
        }).toJson();
        const data = await this.rpc.request(svc, "ListDispatch", req, await this.authHeader({
            roomAdmin: true,
            room: roomName
        }));
        const res = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListAgentDispatchResponse"].fromJson(data, {
            ignoreUnknownFields: true
        });
        if (res.agentDispatches.length === 0) {
            return void 0;
        }
        return res.agentDispatches[0];
    }
    /**
   * List all agent dispatches for a room
   * @param roomName - name of the room to list dispatches for
   * @returns the list of dispatches
   */ async listDispatch(roomName) {
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListAgentDispatchRequest"]({
            room: roomName
        }).toJson();
        const data = await this.rpc.request(svc, "ListDispatch", req, await this.authHeader({
            roomAdmin: true,
            room: roomName
        }));
        const res = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListAgentDispatchResponse"].fromJson(data, {
            ignoreUnknownFields: true
        });
        return res.agentDispatches;
    }
}
;
 //# sourceMappingURL=AgentDispatchClient.js.map
}}),
"[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/EgressClient.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "EgressClient": (()=>EgressClient)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$module__evaluation$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+protocol@1.39.2/node_modules/@livekit/protocol/dist/index.mjs [app-route] (ecmascript) <module evaluation>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+protocol@1.39.2/node_modules/@livekit/protocol/dist/index.mjs [app-route] (ecmascript) <locals>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$ServiceBase$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/ServiceBase.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$TwirpRPC$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/TwirpRPC.js [app-route] (ecmascript)");
;
;
;
const svc = "Egress";
class EgressClient extends __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$ServiceBase$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ServiceBase"] {
    /**
   * @param host - hostname including protocol. i.e. 'https://<project>.livekit.cloud'
   * @param apiKey - API Key, can be set in env var LIVEKIT_API_KEY
   * @param secret - API Secret, can be set in env var LIVEKIT_API_SECRET
   */ constructor(host, apiKey, secret){
        super(apiKey, secret);
        this.rpc = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$TwirpRPC$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["TwirpRpc"](host, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$TwirpRPC$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["livekitPackage"]);
    }
    async startRoomCompositeEgress(roomName, output, optsOrLayout, options, audioOnly, videoOnly, customBaseUrl, audioMixing) {
        let layout;
        if (optsOrLayout !== void 0) {
            if (typeof optsOrLayout === "string") {
                layout = optsOrLayout;
            } else {
                const opts = optsOrLayout;
                layout = opts.layout;
                options = opts.encodingOptions;
                audioOnly = opts.audioOnly;
                videoOnly = opts.videoOnly;
                customBaseUrl = opts.customBaseUrl;
                audioMixing = opts.audioMixing;
            }
        }
        layout ??= "";
        audioOnly ??= false;
        videoOnly ??= false;
        customBaseUrl ??= "";
        audioMixing ??= __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["AudioMixing"].DEFAULT_MIXING;
        const { output: legacyOutput, options: egressOptions, fileOutputs, streamOutputs, segmentOutputs, imageOutputs } = this.getOutputParams(output, options);
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["RoomCompositeEgressRequest"]({
            roomName,
            layout,
            audioOnly,
            audioMixing,
            videoOnly,
            customBaseUrl,
            output: legacyOutput,
            options: egressOptions,
            fileOutputs,
            streamOutputs,
            segmentOutputs,
            imageOutputs
        }).toJson();
        const data = await this.rpc.request(svc, "StartRoomCompositeEgress", req, await this.authHeader({
            roomRecord: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["EgressInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * @param url - url
   * @param output - file or stream output
   * @param opts - WebOptions
   */ async startWebEgress(url, output, opts) {
        const audioOnly = (opts == null ? void 0 : opts.audioOnly) || false;
        const videoOnly = (opts == null ? void 0 : opts.videoOnly) || false;
        const awaitStartSignal = (opts == null ? void 0 : opts.awaitStartSignal) || false;
        const { output: legacyOutput, options, fileOutputs, streamOutputs, segmentOutputs, imageOutputs } = this.getOutputParams(output, opts == null ? void 0 : opts.encodingOptions);
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["WebEgressRequest"]({
            url,
            audioOnly,
            videoOnly,
            awaitStartSignal,
            output: legacyOutput,
            options,
            fileOutputs,
            streamOutputs,
            segmentOutputs,
            imageOutputs
        }).toJson();
        const data = await this.rpc.request(svc, "StartWebEgress", req, await this.authHeader({
            roomRecord: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["EgressInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * Export a participant's audio and video tracks,
   *
   * @param roomName - room name
   * @param output - one or more outputs
   * @param opts - ParticipantEgressOptions
   */ async startParticipantEgress(roomName, identity, output, opts) {
        const { options, fileOutputs, streamOutputs, segmentOutputs, imageOutputs } = this.getOutputParams(output, opts == null ? void 0 : opts.encodingOptions);
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ParticipantEgressRequest"]({
            roomName,
            identity,
            screenShare: (opts == null ? void 0 : opts.screenShare) ?? false,
            options,
            fileOutputs,
            streamOutputs,
            segmentOutputs,
            imageOutputs
        }).toJson();
        const data = await this.rpc.request(svc, "StartParticipantEgress", req, await this.authHeader({
            roomRecord: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["EgressInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    async startTrackCompositeEgress(roomName, output, optsOrAudioTrackId, videoTrackId, options) {
        let audioTrackId;
        if (optsOrAudioTrackId !== void 0) {
            if (typeof optsOrAudioTrackId === "string") {
                audioTrackId = optsOrAudioTrackId;
            } else {
                const opts = optsOrAudioTrackId;
                audioTrackId = opts.audioTrackId;
                videoTrackId = opts.videoTrackId;
                options = opts.encodingOptions;
            }
        }
        audioTrackId ??= "";
        videoTrackId ??= "";
        const { output: legacyOutput, options: egressOptions, fileOutputs, streamOutputs, segmentOutputs, imageOutputs } = this.getOutputParams(output, options);
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["TrackCompositeEgressRequest"]({
            roomName,
            audioTrackId,
            videoTrackId,
            output: legacyOutput,
            options: egressOptions,
            fileOutputs,
            streamOutputs,
            segmentOutputs,
            imageOutputs
        }).toJson();
        const data = await this.rpc.request(svc, "StartTrackCompositeEgress", req, await this.authHeader({
            roomRecord: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["EgressInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    isEncodedOutputs(output) {
        return output.file !== void 0 || output.stream !== void 0 || output.segments !== void 0 || output.images !== void 0;
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    isEncodedFileOutput(output) {
        return output.filepath !== void 0 || output.fileType !== void 0;
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    isSegmentedFileOutput(output) {
        return output.filenamePrefix !== void 0 || output.playlistName !== void 0 || output.filenameSuffix !== void 0;
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    isStreamOutput(output) {
        return output.protocol !== void 0 || output.urls !== void 0;
    }
    getOutputParams(output, opts) {
        let file;
        let fileOutputs;
        let stream;
        let streamOutputs;
        let segments;
        let segmentOutputs;
        let imageOutputs;
        if (this.isEncodedOutputs(output)) {
            if (output.file !== void 0) {
                fileOutputs = [
                    output.file
                ];
            }
            if (output.stream !== void 0) {
                streamOutputs = [
                    output.stream
                ];
            }
            if (output.segments !== void 0) {
                segmentOutputs = [
                    output.segments
                ];
            }
            if (output.images !== void 0) {
                imageOutputs = [
                    output.images
                ];
            }
        } else if (this.isEncodedFileOutput(output)) {
            file = output;
            fileOutputs = [
                file
            ];
        } else if (this.isSegmentedFileOutput(output)) {
            segments = output;
            segmentOutputs = [
                segments
            ];
        } else if (this.isStreamOutput(output)) {
            stream = output;
            streamOutputs = [
                stream
            ];
        }
        let legacyOutput;
        if (file) {
            legacyOutput = {
                case: "file",
                value: file
            };
        } else if (stream) {
            legacyOutput = {
                case: "stream",
                value: stream
            };
        } else if (segments) {
            legacyOutput = {
                case: "segments",
                value: segments
            };
        }
        let egressOptions;
        if (opts) {
            if (typeof opts === "number") {
                egressOptions = {
                    case: "preset",
                    value: opts
                };
            } else {
                egressOptions = {
                    case: "advanced",
                    value: opts
                };
            }
        }
        return {
            output: legacyOutput,
            options: egressOptions,
            fileOutputs,
            streamOutputs,
            segmentOutputs,
            imageOutputs
        };
    }
    /**
   * @param roomName - room name
   * @param output - file or websocket output
   * @param trackId - track Id
   */ async startTrackEgress(roomName, output, trackId) {
        let legacyOutput;
        if (typeof output === "string") {
            legacyOutput = {
                case: "websocketUrl",
                value: output
            };
        } else {
            legacyOutput = {
                case: "file",
                value: output
            };
        }
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["TrackEgressRequest"]({
            roomName,
            trackId,
            output: legacyOutput
        }).toJson();
        const data = await this.rpc.request(svc, "StartTrackEgress", req, await this.authHeader({
            roomRecord: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["EgressInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * @param egressId -
   * @param layout -
   */ async updateLayout(egressId, layout) {
        const data = await this.rpc.request(svc, "UpdateLayout", new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["UpdateLayoutRequest"]({
            egressId,
            layout
        }).toJson(), await this.authHeader({
            roomRecord: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["EgressInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * @param egressId -
   * @param addOutputUrls -
   * @param removeOutputUrls -
   */ async updateStream(egressId, addOutputUrls, removeOutputUrls) {
        addOutputUrls ??= [];
        removeOutputUrls ??= [];
        const data = await this.rpc.request(svc, "UpdateStream", new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["UpdateStreamRequest"]({
            egressId,
            addOutputUrls,
            removeOutputUrls
        }).toJson(), await this.authHeader({
            roomRecord: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["EgressInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * @param roomName - list egress for one room only
   */ async listEgress(options) {
        let req = {};
        if (typeof options === "string") {
            req.roomName = options;
        } else if (options !== void 0) {
            req = options;
        }
        const data = await this.rpc.request(svc, "ListEgress", new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListEgressRequest"](req).toJson(), await this.authHeader({
            roomRecord: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListEgressResponse"].fromJson(data, {
            ignoreUnknownFields: true
        }).items ?? [];
    }
    /**
   * @param egressId -
   */ async stopEgress(egressId) {
        const data = await this.rpc.request(svc, "StopEgress", new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["StopEgressRequest"]({
            egressId
        }).toJson(), await this.authHeader({
            roomRecord: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["EgressInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
}
;
 //# sourceMappingURL=EgressClient.js.map
}}),
"[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/IngressClient.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "IngressClient": (()=>IngressClient)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$module__evaluation$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+protocol@1.39.2/node_modules/@livekit/protocol/dist/index.mjs [app-route] (ecmascript) <module evaluation>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+protocol@1.39.2/node_modules/@livekit/protocol/dist/index.mjs [app-route] (ecmascript) <locals>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$ServiceBase$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/ServiceBase.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$TwirpRPC$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/TwirpRPC.js [app-route] (ecmascript)");
;
;
;
const svc = "Ingress";
class IngressClient extends __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$ServiceBase$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ServiceBase"] {
    /**
   * @param host - hostname including protocol. i.e. 'https://<project>.livekit.cloud'
   * @param apiKey - API Key, can be set in env var LIVEKIT_API_KEY
   * @param secret - API Secret, can be set in env var LIVEKIT_API_SECRET
   */ constructor(host, apiKey, secret){
        super(apiKey, secret);
        this.rpc = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$TwirpRPC$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["TwirpRpc"](host, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$TwirpRPC$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["livekitPackage"]);
    }
    /**
   * @param inputType - protocol for the ingress
   * @param opts - CreateIngressOptions
   */ async createIngress(inputType, opts) {
        let name = "";
        let participantName = "";
        let participantIdentity = "";
        let bypassTranscoding = false;
        let url = "";
        if (opts == null) {
            throw new Error("options dictionary is required");
        }
        const roomName = opts.roomName;
        const enableTranscoding = opts.enableTranscoding;
        const audio = opts.audio;
        const video = opts.video;
        const participantMetadata = opts.participantMetadata;
        name = opts.name || "";
        participantName = opts.participantName || "";
        participantIdentity = opts.participantIdentity || "";
        bypassTranscoding = opts.bypassTranscoding || false;
        url = opts.url || "";
        if (typeof roomName == "undefined") {
            throw new Error("required roomName option not provided");
        }
        if (participantIdentity == "") {
            throw new Error("required participantIdentity option not provided");
        }
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["CreateIngressRequest"]({
            inputType,
            name,
            roomName,
            participantIdentity,
            participantMetadata,
            participantName,
            bypassTranscoding,
            enableTranscoding,
            url,
            audio,
            video
        }).toJson();
        const data = await this.rpc.request(svc, "CreateIngress", req, await this.authHeader({
            ingressAdmin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["IngressInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * @param ingressId - ID of the ingress to update
   * @param opts - UpdateIngressOptions
   */ async updateIngress(ingressId, opts) {
        const name = opts.name || "";
        const roomName = opts.roomName || "";
        const participantName = opts.participantName || "";
        const participantIdentity = opts.participantIdentity || "";
        const { participantMetadata } = opts;
        const { audio, video, bypassTranscoding, enableTranscoding } = opts;
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["UpdateIngressRequest"]({
            ingressId,
            name,
            roomName,
            participantIdentity,
            participantName,
            participantMetadata,
            bypassTranscoding,
            enableTranscoding,
            audio,
            video
        }).toJson();
        const data = await this.rpc.request(svc, "UpdateIngress", req, await this.authHeader({
            ingressAdmin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["IngressInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * @param arg - list room name or options
   */ async listIngress(arg) {
        let req = {};
        if (typeof arg === "string") {
            req.roomName = arg;
        } else if (arg) {
            req = arg;
        }
        const data = await this.rpc.request(svc, "ListIngress", new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListIngressRequest"](req).toJson(), await this.authHeader({
            ingressAdmin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListIngressResponse"].fromJson(data, {
            ignoreUnknownFields: true
        }).items ?? [];
    }
    /**
   * @param ingressId - ingress to delete
   */ async deleteIngress(ingressId) {
        const data = await this.rpc.request(svc, "DeleteIngress", new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["DeleteIngressRequest"]({
            ingressId
        }).toJson(), await this.authHeader({
            ingressAdmin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["IngressInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
}
;
 //# sourceMappingURL=IngressClient.js.map
}}),
"[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/crypto/uuid.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "getRandomBytes": (()=>getRandomBytes)
});
async function getRandomBytes(size = 16) {
    if (globalThis.crypto) {
        return crypto.getRandomValues(new Uint8Array(size));
    } else {
        const nodeCrypto = await __turbopack_context__.r("[externals]/node:crypto [external] (node:crypto, cjs, async loader)")(__turbopack_context__.i);
        return nodeCrypto.getRandomValues(new Uint8Array(size));
    }
}
;
 //# sourceMappingURL=uuid.js.map
}}),
"[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/RoomServiceClient.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "RoomServiceClient": (()=>RoomServiceClient)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$module__evaluation$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+protocol@1.39.2/node_modules/@livekit/protocol/dist/index.mjs [app-route] (ecmascript) <module evaluation>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+protocol@1.39.2/node_modules/@livekit/protocol/dist/index.mjs [app-route] (ecmascript) <locals>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$ServiceBase$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/ServiceBase.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$TwirpRPC$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/TwirpRPC.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$crypto$2f$uuid$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/crypto/uuid.js [app-route] (ecmascript)");
;
;
;
;
const svc = "RoomService";
class RoomServiceClient extends __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$ServiceBase$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ServiceBase"] {
    /**
   *
   * @param host - hostname including protocol. i.e. 'https://<project>.livekit.cloud'
   * @param apiKey - API Key, can be set in env var LIVEKIT_API_KEY
   * @param secret - API Secret, can be set in env var LIVEKIT_API_SECRET
   */ constructor(host, apiKey, secret){
        super(apiKey, secret);
        this.rpc = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$TwirpRPC$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["TwirpRpc"](host, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$TwirpRPC$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["livekitPackage"]);
    }
    /**
   * Creates a new room. Explicit room creation is not required, since rooms will
   * be automatically created when the first participant joins. This method can be
   * used to customize room settings.
   * @param options -
   */ async createRoom(options) {
        const data = await this.rpc.request(svc, "CreateRoom", new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["CreateRoomRequest"](options).toJson(), await this.authHeader({
            roomCreate: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["Room"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * List active rooms
   * @param names - when undefined or empty, list all rooms.
   *                otherwise returns rooms with matching names
   * @returns
   */ async listRooms(names) {
        const data = await this.rpc.request(svc, "ListRooms", new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListRoomsRequest"]({
            names: names ?? []
        }).toJson(), await this.authHeader({
            roomList: true
        }));
        const res = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListRoomsResponse"].fromJson(data, {
            ignoreUnknownFields: true
        });
        return res.rooms ?? [];
    }
    async deleteRoom(room) {
        await this.rpc.request(svc, "DeleteRoom", new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["DeleteRoomRequest"]({
            room
        }).toJson(), await this.authHeader({
            roomCreate: true
        }));
    }
    /**
   * Update metadata of a room
   * @param room - name of the room
   * @param metadata - the new metadata for the room
   */ async updateRoomMetadata(room, metadata) {
        const data = await this.rpc.request(svc, "UpdateRoomMetadata", new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["UpdateRoomMetadataRequest"]({
            room,
            metadata
        }).toJson(), await this.authHeader({
            roomAdmin: true,
            room
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["Room"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * List participants in a room
   * @param room - name of the room
   */ async listParticipants(room) {
        const data = await this.rpc.request(svc, "ListParticipants", new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListParticipantsRequest"]({
            room
        }).toJson(), await this.authHeader({
            roomAdmin: true,
            room
        }));
        const res = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListParticipantsResponse"].fromJson(data, {
            ignoreUnknownFields: true
        });
        return res.participants ?? [];
    }
    /**
   * Get information on a specific participant, including the tracks that participant
   * has published
   * @param room - name of the room
   * @param identity - identity of the participant to return
   */ async getParticipant(room, identity) {
        const data = await this.rpc.request(svc, "GetParticipant", new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["RoomParticipantIdentity"]({
            room,
            identity
        }).toJson(), await this.authHeader({
            roomAdmin: true,
            room
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ParticipantInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * Removes a participant in the room. This will disconnect the participant
   * and will emit a Disconnected event for that participant.
   * Even after being removed, the participant can still re-join the room.
   * @param room -
   * @param identity -
   */ async removeParticipant(room, identity) {
        await this.rpc.request(svc, "RemoveParticipant", new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["RoomParticipantIdentity"]({
            room,
            identity
        }).toJson(), await this.authHeader({
            roomAdmin: true,
            room
        }));
    }
    /**
   * Forwards a participant's track to another room. This will create a
   * participant to join the destination room that has same information
   * with the source participant except the kind to be `Forwarded`. All
   * changes to the source participant will be reflected to the forwarded
   * participant. When the source participant disconnects or the
   * `RemoveParticipant` method is called in the destination room, the
   * forwarding will be stopped.
   * @param room -
   * @param identity -
   * @param destinationRoom - the room to forward the participant to
   */ async forwardParticipant(room, identity, destinationRoom) {
        await this.rpc.request(svc, "ForwardParticipant", new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ForwardParticipantRequest"]({
            room,
            identity,
            destinationRoom
        }).toJson(), await this.authHeader({
            roomAdmin: true,
            room,
            destinationRoom
        }));
    }
    /**
   * Move a connected participant to a different room. Requires `roomAdmin` and `destinationRoom`.
   * The participant will be removed from the current room and added to the destination room.
   * From the other observers' perspective, the participant would've disconnected from the previous room and joined the new one.
   * @param room -
   * @param identity -
   * @param destinationRoom - the room to move the participant to
   */ async moveParticipant(room, identity, destinationRoom) {
        await this.rpc.request(svc, "MoveParticipant", new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["MoveParticipantRequest"]({
            room,
            identity,
            destinationRoom
        }).toJson(), await this.authHeader({
            roomAdmin: true,
            room,
            destinationRoom
        }));
    }
    /**
   * Mutes a track that the participant has published.
   * @param room -
   * @param identity -
   * @param trackSid - sid of the track to be muted
   * @param muted - true to mute, false to unmute
   */ async mutePublishedTrack(room, identity, trackSid, muted) {
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["MuteRoomTrackRequest"]({
            room,
            identity,
            trackSid,
            muted
        }).toJson();
        const data = await this.rpc.request(svc, "MutePublishedTrack", req, await this.authHeader({
            roomAdmin: true,
            room
        }));
        const res = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["MuteRoomTrackResponse"].fromJson(data, {
            ignoreUnknownFields: true
        });
        return res.track;
    }
    async updateParticipant(room, identity, metadataOrOptions, maybePermission, maybeName) {
        const hasOptions = typeof metadataOrOptions === "object";
        const metadata = hasOptions ? metadataOrOptions == null ? void 0 : metadataOrOptions.metadata : metadataOrOptions;
        const permission = hasOptions ? metadataOrOptions.permission : maybePermission;
        const name = hasOptions ? metadataOrOptions.name : maybeName;
        const attributes = hasOptions ? metadataOrOptions.attributes : {};
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["UpdateParticipantRequest"]({
            room,
            identity,
            attributes,
            metadata,
            name
        });
        if (permission) {
            req.permission = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ParticipantPermission"](permission);
        }
        const data = await this.rpc.request(svc, "UpdateParticipant", req.toJson(), await this.authHeader({
            roomAdmin: true,
            room
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ParticipantInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * Updates a participant's subscription to tracks
   * @param room -
   * @param identity -
   * @param trackSids -
   * @param subscribe - true to subscribe, false to unsubscribe
   */ async updateSubscriptions(room, identity, trackSids, subscribe) {
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["UpdateSubscriptionsRequest"]({
            room,
            identity,
            trackSids,
            subscribe,
            participantTracks: []
        }).toJson();
        await this.rpc.request(svc, "UpdateSubscriptions", req, await this.authHeader({
            roomAdmin: true,
            room
        }));
    }
    async sendData(room, data, kind, options = {}) {
        const destinationSids = Array.isArray(options) ? options : options.destinationSids;
        const topic = Array.isArray(options) ? void 0 : options.topic;
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SendDataRequest"]({
            room,
            data,
            kind,
            destinationSids: destinationSids ?? [],
            topic
        });
        if (!Array.isArray(options) && options.destinationIdentities) {
            req.destinationIdentities = options.destinationIdentities;
        }
        req.nonce = await (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$crypto$2f$uuid$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["getRandomBytes"])(16);
        await this.rpc.request(svc, "SendData", req.toJson(), await this.authHeader({
            roomAdmin: true,
            room
        }));
    }
}
;
 //# sourceMappingURL=RoomServiceClient.js.map
}}),
"[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/SipClient.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "SipClient": (()=>SipClient)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$google$2f$protobuf$2f$duration_pb$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/google/protobuf/duration_pb.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$module__evaluation$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+protocol@1.39.2/node_modules/@livekit/protocol/dist/index.mjs [app-route] (ecmascript) <module evaluation>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+protocol@1.39.2/node_modules/@livekit/protocol/dist/index.mjs [app-route] (ecmascript) <locals>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$ServiceBase$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/ServiceBase.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$TwirpRPC$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/TwirpRPC.js [app-route] (ecmascript)");
;
;
;
;
const svc = "SIP";
class SipClient extends __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$ServiceBase$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ServiceBase"] {
    /**
   * @param host - hostname including protocol. i.e. 'https://<project>.livekit.cloud'
   * @param apiKey - API Key, can be set in env var LIVEKIT_API_KEY
   * @param secret - API Secret, can be set in env var LIVEKIT_API_SECRET
   */ constructor(host, apiKey, secret){
        super(apiKey, secret);
        this.rpc = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$TwirpRPC$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["TwirpRpc"](host, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$TwirpRPC$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["livekitPackage"]);
    }
    /**
   * @param number - phone number of the trunk
   * @param opts - CreateSipTrunkOptions
   * @deprecated use `createSipInboundTrunk` or `createSipOutboundTrunk`
   */ async createSipTrunk(number, opts) {
        let inboundAddresses;
        let inboundNumbers;
        let inboundUsername = "";
        let inboundPassword = "";
        let outboundAddress = "";
        let outboundUsername = "";
        let outboundPassword = "";
        let name = "";
        let metadata = "";
        if (opts !== void 0) {
            inboundAddresses = opts.inbound_addresses;
            inboundNumbers = opts.inbound_numbers;
            inboundUsername = opts.inbound_username || "";
            inboundPassword = opts.inbound_password || "";
            outboundAddress = opts.outbound_address || "";
            outboundUsername = opts.outbound_username || "";
            outboundPassword = opts.outbound_password || "";
            name = opts.name || "";
            metadata = opts.metadata || "";
        }
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["CreateSIPTrunkRequest"]({
            name,
            metadata,
            inboundAddresses,
            inboundNumbers,
            inboundUsername,
            inboundPassword,
            outboundNumber: number,
            outboundAddress,
            outboundUsername,
            outboundPassword
        }).toJson();
        const data = await this.rpc.request(svc, "CreateSIPTrunk", req, await this.authHeader({}, {
            admin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPTrunkInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * Create a new SIP inbound trunk.
   *
   * @param name - human-readable name of the trunk
   * @param numbers - phone numbers of the trunk
   * @param opts - CreateSipTrunkOptions
   * @returns Created SIP inbound trunk
   */ async createSipInboundTrunk(name, numbers, opts) {
        if (opts === void 0) {
            opts = {};
        }
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["CreateSIPInboundTrunkRequest"]({
            trunk: new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPInboundTrunkInfo"]({
                name,
                numbers,
                metadata: opts == null ? void 0 : opts.metadata,
                allowedAddresses: opts.allowedAddresses ?? opts.allowed_addresses,
                allowedNumbers: opts.allowedNumbers ?? opts.allowed_numbers,
                authUsername: opts.authUsername ?? opts.auth_username,
                authPassword: opts.authPassword ?? opts.auth_password,
                headers: opts.headers,
                headersToAttributes: opts.headersToAttributes,
                includeHeaders: opts.includeHeaders,
                krispEnabled: opts.krispEnabled
            })
        }).toJson();
        const data = await this.rpc.request(svc, "CreateSIPInboundTrunk", req, await this.authHeader({}, {
            admin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPInboundTrunkInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * Create a new SIP outbound trunk.
   *
   * @param name - human-readable name of the trunk
   * @param address - hostname and port of the SIP server to dial
   * @param numbers - phone numbers of the trunk
   * @param opts - CreateSipTrunkOptions
   * @returns Created SIP outbound trunk
   */ async createSipOutboundTrunk(name, address, numbers, opts) {
        if (opts === void 0) {
            opts = {
                transport: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPTransport"].SIP_TRANSPORT_AUTO
            };
        }
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["CreateSIPOutboundTrunkRequest"]({
            trunk: new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPOutboundTrunkInfo"]({
                name,
                address,
                numbers,
                metadata: opts.metadata,
                transport: opts.transport,
                authUsername: opts.authUsername ?? opts.auth_username,
                authPassword: opts.authPassword ?? opts.auth_password,
                headers: opts.headers,
                headersToAttributes: opts.headersToAttributes,
                includeHeaders: opts.includeHeaders,
                destinationCountry: opts.destinationCountry
            })
        }).toJson();
        const data = await this.rpc.request(svc, "CreateSIPOutboundTrunk", req, await this.authHeader({}, {
            admin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPOutboundTrunkInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * @deprecated use `listSipInboundTrunk` or `listSipOutboundTrunk`
   */ async listSipTrunk() {
        const req = {};
        const data = await this.rpc.request(svc, "ListSIPTrunk", new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListSIPTrunkRequest"](req).toJson(), await this.authHeader({}, {
            admin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListSIPTrunkResponse"].fromJson(data, {
            ignoreUnknownFields: true
        }).items ?? [];
    }
    /**
   * List SIP inbound trunks with optional filtering.
   *
   * @param list - Request with optional filtering parameters
   * @returns Response containing list of SIP inbound trunks
   */ async listSipInboundTrunk(list = {}) {
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListSIPInboundTrunkRequest"](list).toJson();
        const data = await this.rpc.request(svc, "ListSIPInboundTrunk", req, await this.authHeader({}, {
            admin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListSIPInboundTrunkResponse"].fromJson(data, {
            ignoreUnknownFields: true
        }).items ?? [];
    }
    /**
   * List SIP outbound trunks with optional filtering.
   *
   * @param list - Request with optional filtering parameters
   * @returns Response containing list of SIP outbound trunks
   */ async listSipOutboundTrunk(list = {}) {
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListSIPOutboundTrunkRequest"](list).toJson();
        const data = await this.rpc.request(svc, "ListSIPOutboundTrunk", req, await this.authHeader({}, {
            admin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListSIPOutboundTrunkResponse"].fromJson(data, {
            ignoreUnknownFields: true
        }).items ?? [];
    }
    /**
   * Delete a SIP trunk.
   *
   * @param sipTrunkId - ID of the SIP trunk to delete
   * @returns Deleted trunk information
   */ async deleteSipTrunk(sipTrunkId) {
        const data = await this.rpc.request(svc, "DeleteSIPTrunk", new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["DeleteSIPTrunkRequest"]({
            sipTrunkId
        }).toJson(), await this.authHeader({}, {
            admin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPTrunkInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * Create a new SIP dispatch rule.
   *
   * @param rule - SIP dispatch rule to create
   * @param opts - CreateSipDispatchRuleOptions
   * @returns Created SIP dispatch rule
   */ async createSipDispatchRule(rule, opts) {
        if (opts === void 0) {
            opts = {};
        }
        let ruleProto = void 0;
        if (rule.type == "direct") {
            ruleProto = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPDispatchRule"]({
                rule: {
                    case: "dispatchRuleDirect",
                    value: new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPDispatchRuleDirect"]({
                        roomName: rule.roomName,
                        pin: rule.pin || ""
                    })
                }
            });
        } else if (rule.type == "individual") {
            ruleProto = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPDispatchRule"]({
                rule: {
                    case: "dispatchRuleIndividual",
                    value: new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPDispatchRuleIndividual"]({
                        roomPrefix: rule.roomPrefix,
                        pin: rule.pin || ""
                    })
                }
            });
        }
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["CreateSIPDispatchRuleRequest"]({
            rule: ruleProto,
            trunkIds: opts.trunkIds,
            hidePhoneNumber: opts.hidePhoneNumber,
            name: opts.name,
            metadata: opts.metadata,
            attributes: opts.attributes,
            roomPreset: opts.roomPreset,
            roomConfig: opts.roomConfig
        }).toJson();
        const data = await this.rpc.request(svc, "CreateSIPDispatchRule", req, await this.authHeader({}, {
            admin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPDispatchRuleInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * Updates an existing SIP dispatch rule by replacing it entirely.
   *
   * @param sipDispatchRuleId - ID of the SIP dispatch rule to update
   * @param rule - new SIP dispatch rule
   * @returns Updated SIP dispatch rule
   */ async updateSipDispatchRule(sipDispatchRuleId, rule) {
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["UpdateSIPDispatchRuleRequest"]({
            sipDispatchRuleId,
            action: {
                case: "replace",
                value: rule
            }
        }).toJson();
        const data = await this.rpc.request(svc, "UpdateSIPDispatchRule", req, await this.authHeader({}, {
            admin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPDispatchRuleInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * Updates specific fields of an existing SIP dispatch rule.
   * Only provided fields will be updated.
   *
   * @param sipDispatchRuleId - ID of the SIP dispatch rule to update
   * @param fields - Fields of the dispatch rule to update
   * @returns Updated SIP dispatch rule
   */ async updateSipDispatchRuleFields(sipDispatchRuleId, fields = {}) {
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["UpdateSIPDispatchRuleRequest"]({
            sipDispatchRuleId,
            action: {
                case: "update",
                value: fields
            }
        }).toJson();
        const data = await this.rpc.request(svc, "UpdateSIPDispatchRule", req, await this.authHeader({}, {
            admin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPDispatchRuleInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * Updates an existing SIP inbound trunk by replacing it entirely.
   *
   * @param sipTrunkId - ID of the SIP inbound trunk to update
   * @param trunk - SIP inbound trunk to update with
   * @returns Updated SIP inbound trunk
   */ async updateSipInboundTrunk(sipTrunkId, trunk) {
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["UpdateSIPInboundTrunkRequest"]({
            sipTrunkId,
            action: {
                case: "replace",
                value: trunk
            }
        }).toJson();
        const data = await this.rpc.request(svc, "UpdateSIPInboundTrunk", req, await this.authHeader({}, {
            admin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPInboundTrunkInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * Updates specific fields of an existing SIP inbound trunk.
   * Only provided fields will be updated.
   *
   * @param sipTrunkId - ID of the SIP inbound trunk to update
   * @param fields - Fields of the inbound trunk to update
   * @returns Updated SIP inbound trunk
   */ async updateSipInboundTrunkFields(sipTrunkId, fields) {
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["UpdateSIPInboundTrunkRequest"]({
            sipTrunkId,
            action: {
                case: "update",
                value: fields
            }
        }).toJson();
        const data = await this.rpc.request(svc, "UpdateSIPInboundTrunk", req, await this.authHeader({}, {
            admin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPInboundTrunkInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * Updates an existing SIP outbound trunk by replacing it entirely.
   *
   * @param sipTrunkId - ID of the SIP outbound trunk to update
   * @param trunk - SIP outbound trunk to update with
   * @returns Updated SIP outbound trunk
   */ async updateSipOutboundTrunk(sipTrunkId, trunk) {
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["UpdateSIPOutboundTrunkRequest"]({
            sipTrunkId,
            action: {
                case: "replace",
                value: trunk
            }
        }).toJson();
        const data = await this.rpc.request(svc, "UpdateSIPOutboundTrunk", req, await this.authHeader({}, {
            admin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPOutboundTrunkInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * Updates specific fields of an existing SIP outbound trunk.
   * Only provided fields will be updated.
   *
   * @param sipTrunkId - ID of the SIP outbound trunk to update
   * @param fields - Fields of the outbound trunk to update
   * @returns Updated SIP outbound trunk
   */ async updateSipOutboundTrunkFields(sipTrunkId, fields) {
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["UpdateSIPOutboundTrunkRequest"]({
            sipTrunkId,
            action: {
                case: "update",
                value: fields
            }
        }).toJson();
        const data = await this.rpc.request(svc, "UpdateSIPOutboundTrunk", req, await this.authHeader({}, {
            admin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPOutboundTrunkInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * List SIP dispatch rules with optional filtering.
   *
   * @param list - Request with optional filtering parameters
   * @returns Response containing list of SIP dispatch rules
   */ async listSipDispatchRule(list = {}) {
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListSIPDispatchRuleRequest"](list).toJson();
        const data = await this.rpc.request(svc, "ListSIPDispatchRule", req, await this.authHeader({}, {
            admin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["ListSIPDispatchRuleResponse"].fromJson(data, {
            ignoreUnknownFields: true
        }).items ?? [];
    }
    /**
   * Delete a SIP dispatch rule.
   *
   * @param sipDispatchRuleId - ID of the SIP dispatch rule to delete
   * @returns Deleted rule information
   */ async deleteSipDispatchRule(sipDispatchRuleId) {
        const data = await this.rpc.request(svc, "DeleteSIPDispatchRule", new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["DeleteSIPDispatchRuleRequest"]({
            sipDispatchRuleId
        }).toJson(), await this.authHeader({}, {
            admin: true
        }));
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPDispatchRuleInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * Create a new SIP participant.
   *
   * @param sipTrunkId - sip trunk to use for the call
   * @param number - number to dial
   * @param roomName - room to attach the call to
   * @param opts - CreateSipParticipantOptions
   * @returns Created SIP participant
   */ async createSipParticipant(sipTrunkId, number, roomName, opts) {
        if (opts === void 0) {
            opts = {};
        }
        if (opts.timeout === void 0) {
            opts.timeout = opts.waitUntilAnswered ? 60 : 10;
        }
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["CreateSIPParticipantRequest"]({
            sipTrunkId,
            sipCallTo: number,
            sipNumber: opts.fromNumber,
            roomName,
            participantIdentity: opts.participantIdentity || "sip-participant",
            participantName: opts.participantName,
            participantMetadata: opts.participantMetadata,
            participantAttributes: opts.participantAttributes,
            dtmf: opts.dtmf,
            playDialtone: opts.playDialtone ?? opts.playRingtone,
            headers: opts.headers,
            hidePhoneNumber: opts.hidePhoneNumber,
            includeHeaders: opts.includeHeaders,
            ringingTimeout: opts.ringingTimeout ? new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$google$2f$protobuf$2f$duration_pb$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["Duration"]({
                seconds: BigInt(opts.ringingTimeout)
            }) : void 0,
            maxCallDuration: opts.maxCallDuration ? new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$google$2f$protobuf$2f$duration_pb$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["Duration"]({
                seconds: BigInt(opts.maxCallDuration)
            }) : void 0,
            krispEnabled: opts.krispEnabled,
            waitUntilAnswered: opts.waitUntilAnswered
        }).toJson();
        const data = await this.rpc.request(svc, "CreateSIPParticipant", req, await this.authHeader({}, {
            call: true
        }), opts.timeout);
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["SIPParticipantInfo"].fromJson(data, {
            ignoreUnknownFields: true
        });
    }
    /**
   * Transfer a SIP participant to a different room.
   *
   * @param roomName - room the SIP participant to transfer is connectd to
   * @param participantIdentity - identity of the SIP participant to transfer
   * @param transferTo - SIP URL to transfer the participant to
   * @param opts - TransferSipParticipantOptions
   */ async transferSipParticipant(roomName, participantIdentity, transferTo, opts) {
        if (opts === void 0) {
            opts = {};
        }
        const req = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["TransferSIPParticipantRequest"]({
            participantIdentity,
            roomName,
            transferTo,
            playDialtone: opts.playDialtone,
            headers: opts.headers
        }).toJson();
        await this.rpc.request(svc, "TransferSIPParticipant", req, await this.authHeader({
            roomAdmin: true,
            room: roomName
        }, {
            call: true
        }));
    }
}
;
 //# sourceMappingURL=SipClient.js.map
}}),
"[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/crypto/digest.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "digest": (()=>digest)
});
async function digest(data) {
    var _a;
    if ((_a = globalThis.crypto) == null ? void 0 : _a.subtle) {
        const encoder = new TextEncoder();
        return crypto.subtle.digest("SHA-256", encoder.encode(data));
    } else {
        const nodeCrypto = await __turbopack_context__.r("[externals]/node:crypto [external] (node:crypto, cjs, async loader)")(__turbopack_context__.i);
        return nodeCrypto.createHash("sha256").update(data).digest();
    }
}
;
 //# sourceMappingURL=digest.js.map
}}),
"[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/WebhookReceiver.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "WebhookEvent": (()=>WebhookEvent),
    "WebhookReceiver": (()=>WebhookReceiver),
    "authorizeHeader": (()=>authorizeHeader)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$module__evaluation$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+protocol@1.39.2/node_modules/@livekit/protocol/dist/index.mjs [app-route] (ecmascript) <module evaluation>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+protocol@1.39.2/node_modules/@livekit/protocol/dist/index.mjs [app-route] (ecmascript) <locals>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$AccessToken$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/AccessToken.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$crypto$2f$digest$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/crypto/digest.js [app-route] (ecmascript)");
;
;
;
const authorizeHeader = "Authorize";
class WebhookEvent extends __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__["WebhookEvent"] {
    constructor(){
        super(...arguments);
        this.event = "";
    }
    static fromBinary(bytes, options) {
        return new WebhookEvent().fromBinary(bytes, options);
    }
    static fromJson(jsonValue, options) {
        return new WebhookEvent().fromJson(jsonValue, options);
    }
    static fromJsonString(jsonString, options) {
        return new WebhookEvent().fromJsonString(jsonString, options);
    }
}
class WebhookReceiver {
    constructor(apiKey, apiSecret){
        this.verifier = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$AccessToken$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["TokenVerifier"](apiKey, apiSecret);
    }
    /**
   * @param body - string of the posted body
   * @param authHeader - `Authorization` header from the request
   * @param skipAuth - true to skip auth validation
   * @param clockTolerance - How much tolerance to allow for checks against the auth header to be skewed from the claims
   * @returns The processed webhook event
   */ async receive(body, authHeader, skipAuth = false, clockTolerance) {
        if (!skipAuth) {
            if (!authHeader) {
                throw new Error("authorization header is empty");
            }
            const claims = await this.verifier.verify(authHeader, clockTolerance);
            const hash = await (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$crypto$2f$digest$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["digest"])(body);
            const hashDecoded = btoa(Array.from(new Uint8Array(hash)).map((v)=>String.fromCharCode(v)).join(""));
            if (claims.sha256 !== hashDecoded) {
                throw new Error("sha256 checksum of body does not match");
            }
        }
        return WebhookEvent.fromJson(JSON.parse(body), {
            ignoreUnknownFields: true
        });
    }
}
;
 //# sourceMappingURL=WebhookReceiver.js.map
}}),
"[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/index.js [app-route] (ecmascript) <locals>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$module__evaluation$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+protocol@1.39.2/node_modules/@livekit/protocol/dist/index.mjs [app-route] (ecmascript) <module evaluation>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$AccessToken$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/AccessToken.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$AgentDispatchClient$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/AgentDispatchClient.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$EgressClient$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/EgressClient.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$grants$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/grants.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$IngressClient$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/IngressClient.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$RoomServiceClient$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/RoomServiceClient.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$SipClient$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/SipClient.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$WebhookReceiver$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/WebhookReceiver.js [app-route] (ecmascript)");
;
;
;
;
;
;
;
;
;
;
 //# sourceMappingURL=index.js.map
}}),
"[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/index.js [app-route] (ecmascript) <module evaluation>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$protocol$40$1$2e$39$2e$2$2f$node_modules$2f40$livekit$2f$protocol$2f$dist$2f$index$2e$mjs__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$module__evaluation$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+protocol@1.39.2/node_modules/@livekit/protocol/dist/index.mjs [app-route] (ecmascript) <module evaluation>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$AccessToken$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/AccessToken.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$AgentDispatchClient$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/AgentDispatchClient.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$EgressClient$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/EgressClient.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$grants$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/grants.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$IngressClient$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/IngressClient.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$RoomServiceClient$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/RoomServiceClient.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$SipClient$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/SipClient.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$WebhookReceiver$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/WebhookReceiver.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$server$2d$sdk$40$2$2e$13$2e$1$2f$node_modules$2f$livekit$2d$server$2d$sdk$2f$dist$2f$index$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__$3c$locals$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-server-sdk@2.13.1/node_modules/livekit-server-sdk/dist/index.js [app-route] (ecmascript) <locals>");
}}),

};

//# sourceMappingURL=57ac8_livekit-server-sdk_dist_871acdeb._.js.map