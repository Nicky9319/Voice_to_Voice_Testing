(globalThis.TURBOPACK = globalThis.TURBOPACK || []).push([typeof document === "object" ? document.currentScript : undefined, {

"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/contexts-CH4CA2DN.mjs [app-client] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "$": (()=>Is),
    "A": (()=>Xo),
    "B": (()=>Lo),
    "C": (()=>lo),
    "D": (()=>ys),
    "E": (()=>as),
    "F": (()=>$s),
    "G": (()=>zo),
    "H": (()=>cs),
    "I": (()=>ls),
    "J": (()=>ts),
    "K": (()=>ns),
    "L": (()=>Io),
    "M": (()=>Jo),
    "N": (()=>xs),
    "O": (()=>Ss),
    "P": (()=>Ts),
    "Q": (()=>gs),
    "R": (()=>Wn),
    "S": (()=>ds),
    "T": (()=>ps),
    "U": (()=>Ki),
    "V": (()=>As),
    "W": (()=>Go),
    "X": (()=>Bt),
    "Y": (()=>Zo),
    "Z": (()=>ws),
    "_": (()=>Ms),
    "a": (()=>Ds),
    "a0": (()=>is),
    "a1": (()=>j),
    "a2": (()=>Os),
    "a3": (()=>Ls),
    "a4": (()=>ks),
    "a5": (()=>Ko),
    "a6": (()=>qo),
    "a7": (()=>fs),
    "a8": (()=>os),
    "a9": (()=>Rs),
    "aa": (()=>ho),
    "ab": (()=>Un),
    "ac": (()=>Nn),
    "ad": (()=>bs),
    "ae": (()=>Ro),
    "af": (()=>Ar),
    "ag": (()=>jn),
    "ah": (()=>Fn),
    "ai": (()=>zs),
    "aj": (()=>_s),
    "ak": (()=>No),
    "al": (()=>Fs),
    "am": (()=>Bo),
    "an": (()=>Wo),
    "ao": (()=>Uo),
    "ap": (()=>jo),
    "aq": (()=>Yo),
    "ar": (()=>Ns),
    "as": (()=>Do),
    "at": (()=>Fo),
    "au": (()=>Vo),
    "av": (()=>Ho),
    "aw": (()=>Ws),
    "ax": (()=>Us),
    "b": (()=>Cs),
    "c": (()=>Bs),
    "d": (()=>ms),
    "e": (()=>es),
    "f": (()=>Vs),
    "g": (()=>mo),
    "h": (()=>vs),
    "i": (()=>js),
    "j": (()=>ko),
    "k": (()=>Es),
    "l": (()=>_),
    "m": (()=>$o),
    "n": (()=>Yi),
    "o": (()=>po),
    "p": (()=>N),
    "q": (()=>ss),
    "r": (()=>ao),
    "s": (()=>Ps),
    "t": (()=>In),
    "u": (()=>Hs),
    "v": (()=>us),
    "w": (()=>_o),
    "x": (()=>rs),
    "y": (()=>hs),
    "z": (()=>Qo)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-client@2.15.2_@types+dom-mediacapture-record@1.0.22/node_modules/livekit-client/dist/livekit-client.esm.mjs [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
;
;
const De = Math.min, ae = Math.max, $e = Math.round, Le = Math.floor, Q = (e)=>({
        x: e,
        y: e
    }), Yn = {
    left: "right",
    right: "left",
    bottom: "top",
    top: "bottom"
}, qn = {
    start: "end",
    end: "start"
};
function Et(e, t, n) {
    return ae(e, De(t, n));
}
function He(e, t) {
    return typeof e == "function" ? e(t) : e;
}
function ce(e) {
    return e.split("-")[0];
}
function ze(e) {
    return e.split("-")[1];
}
function qt(e) {
    return e === "x" ? "y" : "x";
}
function Kt(e) {
    return e === "y" ? "height" : "width";
}
function ve(e) {
    return [
        "top",
        "bottom"
    ].includes(ce(e)) ? "y" : "x";
}
function Gt(e) {
    return qt(ve(e));
}
function Kn(e, t, n) {
    n === void 0 && (n = !1);
    const r = ze(e), i = Gt(e), o = Kt(i);
    let s = i === "x" ? r === (n ? "end" : "start") ? "right" : "left" : r === "start" ? "bottom" : "top";
    return t.reference[o] > t.floating[o] && (s = Ne(s)), [
        s,
        Ne(s)
    ];
}
function Gn(e) {
    const t = Ne(e);
    return [
        it(e),
        t,
        it(t)
    ];
}
function it(e) {
    return e.replace(/start|end/g, (t)=>qn[t]);
}
function Qn(e, t, n) {
    const r = [
        "left",
        "right"
    ], i = [
        "right",
        "left"
    ], o = [
        "top",
        "bottom"
    ], s = [
        "bottom",
        "top"
    ];
    switch(e){
        case "top":
        case "bottom":
            return n ? t ? i : r : t ? r : i;
        case "left":
        case "right":
            return t ? o : s;
        default:
            return [];
    }
}
function Jn(e, t, n, r) {
    const i = ze(e);
    let o = Qn(ce(e), n === "start", r);
    return i && (o = o.map((s)=>s + "-" + i), t && (o = o.concat(o.map(it)))), o;
}
function Ne(e) {
    return e.replace(/left|right|bottom|top/g, (t)=>Yn[t]);
}
function Xn(e) {
    return {
        top: 0,
        right: 0,
        bottom: 0,
        left: 0,
        ...e
    };
}
function Zn(e) {
    return typeof e != "number" ? Xn(e) : {
        top: e,
        right: e,
        bottom: e,
        left: e
    };
}
function Fe(e) {
    const { x: t, y: n, width: r, height: i } = e;
    return {
        width: r,
        height: i,
        top: n,
        left: t,
        right: t + r,
        bottom: n + i,
        x: t,
        y: n
    };
}
function Ct(e, t, n) {
    let { reference: r, floating: i } = e;
    const o = ve(t), s = Gt(t), a = Kt(s), c = ce(t), u = o === "y", l = r.x + r.width / 2 - i.width / 2, f = r.y + r.height / 2 - i.height / 2, v = r[a] / 2 - i[a] / 2;
    let d;
    switch(c){
        case "top":
            d = {
                x: l,
                y: r.y - i.height
            };
            break;
        case "bottom":
            d = {
                x: l,
                y: r.y + r.height
            };
            break;
        case "right":
            d = {
                x: r.x + r.width,
                y: f
            };
            break;
        case "left":
            d = {
                x: r.x - i.width,
                y: f
            };
            break;
        default:
            d = {
                x: r.x,
                y: r.y
            };
    }
    switch(ze(t)){
        case "start":
            d[s] -= v * (n && u ? -1 : 1);
            break;
        case "end":
            d[s] += v * (n && u ? -1 : 1);
            break;
    }
    return d;
}
const er = async (e, t, n)=>{
    const { placement: r = "bottom", strategy: i = "absolute", middleware: o = [], platform: s } = n, a = o.filter(Boolean), c = await (s.isRTL == null ? void 0 : s.isRTL(t));
    let u = await s.getElementRects({
        reference: e,
        floating: t,
        strategy: i
    }), { x: l, y: f } = Ct(u, r, c), v = r, d = {}, m = 0;
    for(let p = 0; p < a.length; p++){
        const { name: g, fn: h } = a[p], { x, y: E, data: P, reset: b } = await h({
            x: l,
            y: f,
            initialPlacement: r,
            placement: v,
            strategy: i,
            middlewareData: d,
            rects: u,
            platform: s,
            elements: {
                reference: e,
                floating: t
            }
        });
        l = x ?? l, f = E ?? f, d = {
            ...d,
            [g]: {
                ...d[g],
                ...P
            }
        }, b && m <= 50 && (m++, typeof b == "object" && (b.placement && (v = b.placement), b.rects && (u = b.rects === !0 ? await s.getElementRects({
            reference: e,
            floating: t,
            strategy: i
        }) : b.rects), { x: l, y: f } = Ct(u, v, c)), p = -1);
    }
    return {
        x: l,
        y: f,
        placement: v,
        strategy: i,
        middlewareData: d
    };
};
async function Qt(e, t) {
    var n;
    t === void 0 && (t = {});
    const { x: r, y: i, platform: o, rects: s, elements: a, strategy: c } = e, { boundary: u = "clippingAncestors", rootBoundary: l = "viewport", elementContext: f = "floating", altBoundary: v = !1, padding: d = 0 } = He(t, e), m = Zn(d), g = a[v ? f === "floating" ? "reference" : "floating" : f], h = Fe(await o.getClippingRect({
        element: (n = await (o.isElement == null ? void 0 : o.isElement(g))) == null || n ? g : g.contextElement || await (o.getDocumentElement == null ? void 0 : o.getDocumentElement(a.floating)),
        boundary: u,
        rootBoundary: l,
        strategy: c
    })), x = f === "floating" ? {
        x: r,
        y: i,
        width: s.floating.width,
        height: s.floating.height
    } : s.reference, E = await (o.getOffsetParent == null ? void 0 : o.getOffsetParent(a.floating)), P = await (o.isElement == null ? void 0 : o.isElement(E)) ? await (o.getScale == null ? void 0 : o.getScale(E)) || {
        x: 1,
        y: 1
    } : {
        x: 1,
        y: 1
    }, b = Fe(o.convertOffsetParentRelativeRectToViewportRelativeRect ? await o.convertOffsetParentRelativeRectToViewportRelativeRect({
        elements: a,
        rect: x,
        offsetParent: E,
        strategy: c
    }) : x);
    return {
        top: (h.top - b.top + m.top) / P.y,
        bottom: (b.bottom - h.bottom + m.bottom) / P.y,
        left: (h.left - b.left + m.left) / P.x,
        right: (b.right - h.right + m.right) / P.x
    };
}
const tr = function(e) {
    return e === void 0 && (e = {}), {
        name: "flip",
        options: e,
        async fn (t) {
            var n, r;
            const { placement: i, middlewareData: o, rects: s, initialPlacement: a, platform: c, elements: u } = t, { mainAxis: l = !0, crossAxis: f = !0, fallbackPlacements: v, fallbackStrategy: d = "bestFit", fallbackAxisSideDirection: m = "none", flipAlignment: p = !0, ...g } = He(e, t);
            if ((n = o.arrow) != null && n.alignmentOffset) return {};
            const h = ce(i), x = ve(a), E = ce(a) === a, P = await (c.isRTL == null ? void 0 : c.isRTL(u.floating)), b = v || (E || !p ? [
                Ne(a)
            ] : Gn(a)), S = m !== "none";
            !v && S && b.push(...Jn(a, p, m, P));
            const C = [
                a,
                ...b
            ], $ = await Qt(t, g), I = [];
            let z = ((r = o.flip) == null ? void 0 : r.overflows) || [];
            if (l && I.push($[h]), f) {
                const G = Kn(i, s, P);
                I.push($[G[0]], $[G[1]]);
            }
            if (z = [
                ...z,
                {
                    placement: i,
                    overflows: I
                }
            ], !I.every((G)=>G <= 0)) {
                var T, L;
                const G = (((T = o.flip) == null ? void 0 : T.index) || 0) + 1, ke = C[G];
                if (ke) return {
                    data: {
                        index: G,
                        overflows: z
                    },
                    reset: {
                        placement: ke
                    }
                };
                let Se = (L = z.filter((de)=>de.overflows[0] <= 0).sort((de, ne)=>de.overflows[1] - ne.overflows[1])[0]) == null ? void 0 : L.placement;
                if (!Se) switch(d){
                    case "bestFit":
                        {
                            var Z;
                            const de = (Z = z.filter((ne)=>{
                                if (S) {
                                    const re = ve(ne.placement);
                                    return re === x || // Create a bias to the `y` side axis due to horizontal
                                    // reading directions favoring greater width.
                                    re === "y";
                                }
                                return !0;
                            }).map((ne)=>[
                                    ne.placement,
                                    ne.overflows.filter((re)=>re > 0).reduce((re, Bn)=>re + Bn, 0)
                                ]).sort((ne, re)=>ne[1] - re[1])[0]) == null ? void 0 : Z[0];
                            de && (Se = de);
                            break;
                        }
                    case "initialPlacement":
                        Se = a;
                        break;
                }
                if (i !== Se) return {
                    reset: {
                        placement: Se
                    }
                };
            }
            return {};
        }
    };
};
async function nr(e, t) {
    const { placement: n, platform: r, elements: i } = e, o = await (r.isRTL == null ? void 0 : r.isRTL(i.floating)), s = ce(n), a = ze(n), c = ve(n) === "y", u = [
        "left",
        "top"
    ].includes(s) ? -1 : 1, l = o && c ? -1 : 1, f = He(t, e);
    let { mainAxis: v, crossAxis: d, alignmentAxis: m } = typeof f == "number" ? {
        mainAxis: f,
        crossAxis: 0,
        alignmentAxis: null
    } : {
        mainAxis: f.mainAxis || 0,
        crossAxis: f.crossAxis || 0,
        alignmentAxis: f.alignmentAxis
    };
    return a && typeof m == "number" && (d = a === "end" ? m * -1 : m), c ? {
        x: d * l,
        y: v * u
    } : {
        x: v * u,
        y: d * l
    };
}
const rr = function(e) {
    return e === void 0 && (e = 0), {
        name: "offset",
        options: e,
        async fn (t) {
            var n, r;
            const { x: i, y: o, placement: s, middlewareData: a } = t, c = await nr(t, e);
            return s === ((n = a.offset) == null ? void 0 : n.placement) && (r = a.arrow) != null && r.alignmentOffset ? {} : {
                x: i + c.x,
                y: o + c.y,
                data: {
                    ...c,
                    placement: s
                }
            };
        }
    };
}, ir = function(e) {
    return e === void 0 && (e = {}), {
        name: "shift",
        options: e,
        async fn (t) {
            const { x: n, y: r, placement: i } = t, { mainAxis: o = !0, crossAxis: s = !1, limiter: a = {
                fn: (g)=>{
                    let { x: h, y: x } = g;
                    return {
                        x: h,
                        y: x
                    };
                }
            }, ...c } = He(e, t), u = {
                x: n,
                y: r
            }, l = await Qt(t, c), f = ve(ce(i)), v = qt(f);
            let d = u[v], m = u[f];
            if (o) {
                const g = v === "y" ? "top" : "left", h = v === "y" ? "bottom" : "right", x = d + l[g], E = d - l[h];
                d = Et(x, d, E);
            }
            if (s) {
                const g = f === "y" ? "top" : "left", h = f === "y" ? "bottom" : "right", x = m + l[g], E = m - l[h];
                m = Et(x, m, E);
            }
            const p = a.fn({
                ...t,
                [v]: d,
                [f]: m
            });
            return {
                ...p,
                data: {
                    x: p.x - n,
                    y: p.y - r,
                    enabled: {
                        [v]: o,
                        [f]: s
                    }
                }
            };
        }
    };
};
function Ye() {
    return typeof window < "u";
}
function ye(e) {
    return Jt(e) ? (e.nodeName || "").toLowerCase() : "#document";
}
function U(e) {
    var t;
    return (e == null || (t = e.ownerDocument) == null ? void 0 : t.defaultView) || window;
}
function X(e) {
    var t;
    return (t = (Jt(e) ? e.ownerDocument : e.document) || window.document) == null ? void 0 : t.documentElement;
}
function Jt(e) {
    return Ye() ? e instanceof Node || e instanceof U(e).Node : !1;
}
function q(e) {
    return Ye() ? e instanceof Element || e instanceof U(e).Element : !1;
}
function J(e) {
    return Ye() ? e instanceof HTMLElement || e instanceof U(e).HTMLElement : !1;
}
function Pt(e) {
    return !Ye() || typeof ShadowRoot > "u" ? !1 : e instanceof ShadowRoot || e instanceof U(e).ShadowRoot;
}
function Ae(e) {
    const { overflow: t, overflowX: n, overflowY: r, display: i } = K(e);
    return /auto|scroll|overlay|hidden|clip/.test(t + r + n) && ![
        "inline",
        "contents"
    ].includes(i);
}
function or(e) {
    return [
        "table",
        "td",
        "th"
    ].includes(ye(e));
}
function qe(e) {
    return [
        ":popover-open",
        ":modal"
    ].some((t)=>{
        try {
            return e.matches(t);
        } catch  {
            return !1;
        }
    });
}
function ft(e) {
    const t = dt(), n = q(e) ? K(e) : e;
    return [
        "transform",
        "translate",
        "scale",
        "rotate",
        "perspective"
    ].some((r)=>n[r] ? n[r] !== "none" : !1) || (n.containerType ? n.containerType !== "normal" : !1) || !t && (n.backdropFilter ? n.backdropFilter !== "none" : !1) || !t && (n.filter ? n.filter !== "none" : !1) || [
        "transform",
        "translate",
        "scale",
        "rotate",
        "perspective",
        "filter"
    ].some((r)=>(n.willChange || "").includes(r)) || [
        "paint",
        "layout",
        "strict",
        "content"
    ].some((r)=>(n.contain || "").includes(r));
}
function sr(e) {
    let t = oe(e);
    for(; J(t) && !me(t);){
        if (ft(t)) return t;
        if (qe(t)) return null;
        t = oe(t);
    }
    return null;
}
function dt() {
    return typeof CSS > "u" || !CSS.supports ? !1 : CSS.supports("-webkit-backdrop-filter", "none");
}
function me(e) {
    return [
        "html",
        "body",
        "#document"
    ].includes(ye(e));
}
function K(e) {
    return U(e).getComputedStyle(e);
}
function Ke(e) {
    return q(e) ? {
        scrollLeft: e.scrollLeft,
        scrollTop: e.scrollTop
    } : {
        scrollLeft: e.scrollX,
        scrollTop: e.scrollY
    };
}
function oe(e) {
    if (ye(e) === "html") return e;
    const t = // Step into the shadow DOM of the parent of a slotted node.
    e.assignedSlot || // DOM Element detected.
    e.parentNode || // ShadowRoot detected.
    Pt(e) && e.host || // Fallback.
    X(e);
    return Pt(t) ? t.host : t;
}
function Xt(e) {
    const t = oe(e);
    return me(t) ? e.ownerDocument ? e.ownerDocument.body : e.body : J(t) && Ae(t) ? t : Xt(t);
}
function Ee(e, t, n) {
    var r;
    t === void 0 && (t = []), n === void 0 && (n = !0);
    const i = Xt(e), o = i === ((r = e.ownerDocument) == null ? void 0 : r.body), s = U(i);
    if (o) {
        const a = ot(s);
        return t.concat(s, s.visualViewport || [], Ae(i) ? i : [], a && n ? Ee(a) : []);
    }
    return t.concat(i, Ee(i, [], n));
}
function ot(e) {
    return e.parent && Object.getPrototypeOf(e.parent) ? e.frameElement : null;
}
function Zt(e) {
    const t = K(e);
    let n = parseFloat(t.width) || 0, r = parseFloat(t.height) || 0;
    const i = J(e), o = i ? e.offsetWidth : n, s = i ? e.offsetHeight : r, a = $e(n) !== o || $e(r) !== s;
    return a && (n = o, r = s), {
        width: n,
        height: r,
        $: a
    };
}
function pt(e) {
    return q(e) ? e : e.contextElement;
}
function pe(e) {
    const t = pt(e);
    if (!J(t)) return Q(1);
    const n = t.getBoundingClientRect(), { width: r, height: i, $: o } = Zt(t);
    let s = (o ? $e(n.width) : n.width) / r, a = (o ? $e(n.height) : n.height) / i;
    return (!s || !Number.isFinite(s)) && (s = 1), (!a || !Number.isFinite(a)) && (a = 1), {
        x: s,
        y: a
    };
}
const ar = /* @__PURE__ */ Q(0);
function en(e) {
    const t = U(e);
    return !dt() || !t.visualViewport ? ar : {
        x: t.visualViewport.offsetLeft,
        y: t.visualViewport.offsetTop
    };
}
function cr(e, t, n) {
    return t === void 0 && (t = !1), !n || t && n !== U(e) ? !1 : t;
}
function ue(e, t, n, r) {
    t === void 0 && (t = !1), n === void 0 && (n = !1);
    const i = e.getBoundingClientRect(), o = pt(e);
    let s = Q(1);
    t && (r ? q(r) && (s = pe(r)) : s = pe(e));
    const a = cr(o, n, r) ? en(o) : Q(0);
    let c = (i.left + a.x) / s.x, u = (i.top + a.y) / s.y, l = i.width / s.x, f = i.height / s.y;
    if (o) {
        const v = U(o), d = r && q(r) ? U(r) : r;
        let m = v, p = ot(m);
        for(; p && r && d !== m;){
            const g = pe(p), h = p.getBoundingClientRect(), x = K(p), E = h.left + (p.clientLeft + parseFloat(x.paddingLeft)) * g.x, P = h.top + (p.clientTop + parseFloat(x.paddingTop)) * g.y;
            c *= g.x, u *= g.y, l *= g.x, f *= g.y, c += E, u += P, m = U(p), p = ot(m);
        }
    }
    return Fe({
        width: l,
        height: f,
        x: c,
        y: u
    });
}
function ht(e, t) {
    const n = Ke(e).scrollLeft;
    return t ? t.left + n : ue(X(e)).left + n;
}
function tn(e, t, n) {
    n === void 0 && (n = !1);
    const r = e.getBoundingClientRect(), i = r.left + t.scrollLeft - (n ? 0 : // RTL <body> scrollbar.
    ht(e, r)), o = r.top + t.scrollTop;
    return {
        x: i,
        y: o
    };
}
function ur(e) {
    let { elements: t, rect: n, offsetParent: r, strategy: i } = e;
    const o = i === "fixed", s = X(r), a = t ? qe(t.floating) : !1;
    if (r === s || a && o) return n;
    let c = {
        scrollLeft: 0,
        scrollTop: 0
    }, u = Q(1);
    const l = Q(0), f = J(r);
    if ((f || !f && !o) && ((ye(r) !== "body" || Ae(s)) && (c = Ke(r)), J(r))) {
        const d = ue(r);
        u = pe(r), l.x = d.x + r.clientLeft, l.y = d.y + r.clientTop;
    }
    const v = s && !f && !o ? tn(s, c, !0) : Q(0);
    return {
        width: n.width * u.x,
        height: n.height * u.y,
        x: n.x * u.x - c.scrollLeft * u.x + l.x + v.x,
        y: n.y * u.y - c.scrollTop * u.y + l.y + v.y
    };
}
function lr(e) {
    return Array.from(e.getClientRects());
}
function fr(e) {
    const t = X(e), n = Ke(e), r = e.ownerDocument.body, i = ae(t.scrollWidth, t.clientWidth, r.scrollWidth, r.clientWidth), o = ae(t.scrollHeight, t.clientHeight, r.scrollHeight, r.clientHeight);
    let s = -n.scrollLeft + ht(e);
    const a = -n.scrollTop;
    return K(r).direction === "rtl" && (s += ae(t.clientWidth, r.clientWidth) - i), {
        width: i,
        height: o,
        x: s,
        y: a
    };
}
function dr(e, t) {
    const n = U(e), r = X(e), i = n.visualViewport;
    let o = r.clientWidth, s = r.clientHeight, a = 0, c = 0;
    if (i) {
        o = i.width, s = i.height;
        const u = dt();
        (!u || u && t === "fixed") && (a = i.offsetLeft, c = i.offsetTop);
    }
    return {
        width: o,
        height: s,
        x: a,
        y: c
    };
}
function pr(e, t) {
    const n = ue(e, !0, t === "fixed"), r = n.top + e.clientTop, i = n.left + e.clientLeft, o = J(e) ? pe(e) : Q(1), s = e.clientWidth * o.x, a = e.clientHeight * o.y, c = i * o.x, u = r * o.y;
    return {
        width: s,
        height: a,
        x: c,
        y: u
    };
}
function At(e, t, n) {
    let r;
    if (t === "viewport") r = dr(e, n);
    else if (t === "document") r = fr(X(e));
    else if (q(t)) r = pr(t, n);
    else {
        const i = en(e);
        r = {
            x: t.x - i.x,
            y: t.y - i.y,
            width: t.width,
            height: t.height
        };
    }
    return Fe(r);
}
function nn(e, t) {
    const n = oe(e);
    return n === t || !q(n) || me(n) ? !1 : K(n).position === "fixed" || nn(n, t);
}
function hr(e, t) {
    const n = t.get(e);
    if (n) return n;
    let r = Ee(e, [], !1).filter((a)=>q(a) && ye(a) !== "body"), i = null;
    const o = K(e).position === "fixed";
    let s = o ? oe(e) : e;
    for(; q(s) && !me(s);){
        const a = K(s), c = ft(s);
        !c && a.position === "fixed" && (i = null), (o ? !c && !i : !c && a.position === "static" && !!i && [
            "absolute",
            "fixed"
        ].includes(i.position) || Ae(s) && !c && nn(e, s)) ? r = r.filter((l)=>l !== s) : i = a, s = oe(s);
    }
    return t.set(e, r), r;
}
function vr(e) {
    let { element: t, boundary: n, rootBoundary: r, strategy: i } = e;
    const s = [
        ...n === "clippingAncestors" ? qe(t) ? [] : hr(t, this._c) : [].concat(n),
        r
    ], a = s[0], c = s.reduce((u, l)=>{
        const f = At(t, l, i);
        return u.top = ae(f.top, u.top), u.right = De(f.right, u.right), u.bottom = De(f.bottom, u.bottom), u.left = ae(f.left, u.left), u;
    }, At(t, a, i));
    return {
        width: c.right - c.left,
        height: c.bottom - c.top,
        x: c.left,
        y: c.top
    };
}
function mr(e) {
    const { width: t, height: n } = Zt(e);
    return {
        width: t,
        height: n
    };
}
function gr(e, t, n) {
    const r = J(t), i = X(t), o = n === "fixed", s = ue(e, !0, o, t);
    let a = {
        scrollLeft: 0,
        scrollTop: 0
    };
    const c = Q(0);
    if (r || !r && !o) if ((ye(t) !== "body" || Ae(i)) && (a = Ke(t)), r) {
        const v = ue(t, !0, o, t);
        c.x = v.x + t.clientLeft, c.y = v.y + t.clientTop;
    } else i && (c.x = ht(i));
    const u = i && !r && !o ? tn(i, a) : Q(0), l = s.left + a.scrollLeft - c.x - u.x, f = s.top + a.scrollTop - c.y - u.y;
    return {
        x: l,
        y: f,
        width: s.width,
        height: s.height
    };
}
function Xe(e) {
    return K(e).position === "static";
}
function Ot(e, t) {
    if (!J(e) || K(e).position === "fixed") return null;
    if (t) return t(e);
    let n = e.offsetParent;
    return X(e) === n && (n = n.ownerDocument.body), n;
}
function rn(e, t) {
    const n = U(e);
    if (qe(e)) return n;
    if (!J(e)) {
        let i = oe(e);
        for(; i && !me(i);){
            if (q(i) && !Xe(i)) return i;
            i = oe(i);
        }
        return n;
    }
    let r = Ot(e, t);
    for(; r && or(r) && Xe(r);)r = Ot(r, t);
    return r && me(r) && Xe(r) && !ft(r) ? n : r || sr(e) || n;
}
const br = async function(e) {
    const t = this.getOffsetParent || rn, n = this.getDimensions, r = await n(e.floating);
    return {
        reference: gr(e.reference, await t(e.floating), e.strategy),
        floating: {
            x: 0,
            y: 0,
            width: r.width,
            height: r.height
        }
    };
};
function yr(e) {
    return K(e).direction === "rtl";
}
const wr = {
    convertOffsetParentRelativeRectToViewportRelativeRect: ur,
    getDocumentElement: X,
    getClippingRect: vr,
    getOffsetParent: rn,
    getElementRects: br,
    getClientRects: lr,
    getDimensions: mr,
    getScale: pe,
    isElement: q,
    isRTL: yr
};
function on(e, t) {
    return e.x === t.x && e.y === t.y && e.width === t.width && e.height === t.height;
}
function xr(e, t) {
    let n = null, r;
    const i = X(e);
    function o() {
        var a;
        clearTimeout(r), (a = n) == null || a.disconnect(), n = null;
    }
    function s(a, c) {
        a === void 0 && (a = !1), c === void 0 && (c = 1), o();
        const u = e.getBoundingClientRect(), { left: l, top: f, width: v, height: d } = u;
        if (a || t(), !v || !d) return;
        const m = Le(f), p = Le(i.clientWidth - (l + v)), g = Le(i.clientHeight - (f + d)), h = Le(l), E = {
            rootMargin: -m + "px " + -p + "px " + -g + "px " + -h + "px",
            threshold: ae(0, De(1, c)) || 1
        };
        let P = !0;
        function b(S) {
            const C = S[0].intersectionRatio;
            if (C !== c) {
                if (!P) return s();
                C ? s(!1, C) : r = setTimeout(()=>{
                    s(!1, 1e-7);
                }, 1e3);
            }
            C === 1 && !on(u, e.getBoundingClientRect()) && s(), P = !1;
        }
        try {
            n = new IntersectionObserver(b, {
                ...E,
                // Handle <iframe>s
                root: i.ownerDocument
            });
        } catch  {
            n = new IntersectionObserver(b, E);
        }
        n.observe(e);
    }
    return s(!0), o;
}
function Sr(e, t, n, r) {
    r === void 0 && (r = {});
    const { ancestorScroll: i = !0, ancestorResize: o = !0, elementResize: s = typeof ResizeObserver == "function", layoutShift: a = typeof IntersectionObserver == "function", animationFrame: c = !1 } = r, u = pt(e), l = i || o ? [
        ...u ? Ee(u) : [],
        ...Ee(t)
    ] : [];
    l.forEach((h)=>{
        i && h.addEventListener("scroll", n, {
            passive: !0
        }), o && h.addEventListener("resize", n);
    });
    const f = u && a ? xr(u, n) : null;
    let v = -1, d = null;
    s && (d = new ResizeObserver((h)=>{
        let [x] = h;
        x && x.target === u && d && (d.unobserve(t), cancelAnimationFrame(v), v = requestAnimationFrame(()=>{
            var E;
            (E = d) == null || E.observe(t);
        })), n();
    }), u && !c && d.observe(u), d.observe(t));
    let m, p = c ? ue(e) : null;
    c && g();
    function g() {
        const h = ue(e);
        p && !on(p, h) && n(), p = h, m = requestAnimationFrame(g);
    }
    return n(), ()=>{
        var h;
        l.forEach((x)=>{
            i && x.removeEventListener("scroll", n), o && x.removeEventListener("resize", n);
        }), f == null || f(), (h = d) == null || h.disconnect(), d = null, c && cancelAnimationFrame(m);
    };
}
const Tr = rr, Er = ir, Cr = tr, Pr = (e, t, n)=>{
    const r = /* @__PURE__ */ new Map(), i = {
        platform: wr,
        ...n
    }, o = {
        ...i.platform,
        _c: r
    };
    return er(e, t, {
        ...i,
        platform: o
    });
};
var Ro = typeof globalThis < "u" ? globalThis : typeof window < "u" ? window : typeof global < "u" ? global : typeof self < "u" ? self : {};
function Ar(e) {
    return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var Me = {
    exports: {}
}, Or = Me.exports, kt;
function kr() {
    return kt || (kt = 1, function(e) {
        (function(t, n) {
            e.exports ? e.exports = n() : t.log = n();
        })(Or, function() {
            var t = function() {}, n = "undefined", r = typeof window !== n && typeof window.navigator !== n && /Trident\/|MSIE /.test(window.navigator.userAgent), i = [
                "trace",
                "debug",
                "info",
                "warn",
                "error"
            ], o = {}, s = null;
            function a(p, g) {
                var h = p[g];
                if (typeof h.bind == "function") return h.bind(p);
                try {
                    return Function.prototype.bind.call(h, p);
                } catch  {
                    return function() {
                        return Function.prototype.apply.apply(h, [
                            p,
                            arguments
                        ]);
                    };
                }
            }
            function c() {
                console.log && (console.log.apply ? console.log.apply(console, arguments) : Function.prototype.apply.apply(console.log, [
                    console,
                    arguments
                ])), console.trace && console.trace();
            }
            function u(p) {
                return p === "debug" && (p = "log"), typeof console === n ? !1 : p === "trace" && r ? c : console[p] !== void 0 ? a(console, p) : console.log !== void 0 ? a(console, "log") : t;
            }
            function l() {
                for(var p = this.getLevel(), g = 0; g < i.length; g++){
                    var h = i[g];
                    this[h] = g < p ? t : this.methodFactory(h, p, this.name);
                }
                if (this.log = this.debug, typeof console === n && p < this.levels.SILENT) return "No console available for logging";
            }
            function f(p) {
                return function() {
                    typeof console !== n && (l.call(this), this[p].apply(this, arguments));
                };
            }
            function v(p, g, h) {
                return u(p) || f.apply(this, arguments);
            }
            function d(p, g) {
                var h = this, x, E, P, b = "loglevel";
                typeof p == "string" ? b += ":" + p : typeof p == "symbol" && (b = void 0);
                function S(T) {
                    var L = (i[T] || "silent").toUpperCase();
                    if (!(typeof window === n || !b)) {
                        try {
                            window.localStorage[b] = L;
                            return;
                        } catch  {}
                        try {
                            window.document.cookie = encodeURIComponent(b) + "=" + L + ";";
                        } catch  {}
                    }
                }
                function C() {
                    var T;
                    if (!(typeof window === n || !b)) {
                        try {
                            T = window.localStorage[b];
                        } catch  {}
                        if (typeof T === n) try {
                            var L = window.document.cookie, Z = encodeURIComponent(b), G = L.indexOf(Z + "=");
                            G !== -1 && (T = /^([^;]+)/.exec(L.slice(G + Z.length + 1))[1]);
                        } catch  {}
                        return h.levels[T] === void 0 && (T = void 0), T;
                    }
                }
                function $() {
                    if (!(typeof window === n || !b)) {
                        try {
                            window.localStorage.removeItem(b);
                        } catch  {}
                        try {
                            window.document.cookie = encodeURIComponent(b) + "=; expires=Thu, 01 Jan 1970 00:00:00 UTC";
                        } catch  {}
                    }
                }
                function I(T) {
                    var L = T;
                    if (typeof L == "string" && h.levels[L.toUpperCase()] !== void 0 && (L = h.levels[L.toUpperCase()]), typeof L == "number" && L >= 0 && L <= h.levels.SILENT) return L;
                    throw new TypeError("log.setLevel() called with invalid level: " + T);
                }
                h.name = p, h.levels = {
                    TRACE: 0,
                    DEBUG: 1,
                    INFO: 2,
                    WARN: 3,
                    ERROR: 4,
                    SILENT: 5
                }, h.methodFactory = g || v, h.getLevel = function() {
                    return P ?? E ?? x;
                }, h.setLevel = function(T, L) {
                    return P = I(T), L !== !1 && S(P), l.call(h);
                }, h.setDefaultLevel = function(T) {
                    E = I(T), C() || h.setLevel(T, !1);
                }, h.resetLevel = function() {
                    P = null, $(), l.call(h);
                }, h.enableAll = function(T) {
                    h.setLevel(h.levels.TRACE, T);
                }, h.disableAll = function(T) {
                    h.setLevel(h.levels.SILENT, T);
                }, h.rebuild = function() {
                    if (s !== h && (x = I(s.getLevel())), l.call(h), s === h) for(var T in o)o[T].rebuild();
                }, x = I(s ? s.getLevel() : "WARN");
                var z = C();
                z != null && (P = I(z)), l.call(h);
            }
            s = new d(), s.getLogger = function(g) {
                if (typeof g != "symbol" && typeof g != "string" || g === "") throw new TypeError("You must supply a name when creating a logger.");
                var h = o[g];
                return h || (h = o[g] = new d(g, s.methodFactory)), h;
            };
            var m = typeof window !== n ? window.log : void 0;
            return s.noConflict = function() {
                return typeof window !== n && window.log === s && (window.log = m), s;
            }, s.getLoggers = function() {
                return o;
            }, s.default = s, s;
        });
    }(Me)), Me.exports;
}
var Lr = kr();
const _r = /* @__PURE__ */ Ar(Lr);
var st = function(e, t) {
    return st = Object.setPrototypeOf || ({
        __proto__: []
    }) instanceof Array && function(n, r) {
        n.__proto__ = r;
    } || function(n, r) {
        for(var i in r)Object.prototype.hasOwnProperty.call(r, i) && (n[i] = r[i]);
    }, st(e, t);
};
function te(e, t) {
    if (typeof t != "function" && t !== null) throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
    st(e, t);
    function n() {
        this.constructor = e;
    }
    e.prototype = t === null ? Object.create(t) : (n.prototype = t.prototype, new n());
}
function Ir(e, t, n, r) {
    function i(o) {
        return o instanceof n ? o : new n(function(s) {
            s(o);
        });
    }
    return new (n || (n = Promise))(function(o, s) {
        function a(l) {
            try {
                u(r.next(l));
            } catch (f) {
                s(f);
            }
        }
        function c(l) {
            try {
                u(r.throw(l));
            } catch (f) {
                s(f);
            }
        }
        function u(l) {
            l.done ? o(l.value) : i(l.value).then(a, c);
        }
        u((r = r.apply(e, t || [])).next());
    });
}
function sn(e, t) {
    var n = {
        label: 0,
        sent: function() {
            if (o[0] & 1) throw o[1];
            return o[1];
        },
        trys: [],
        ops: []
    }, r, i, o, s = Object.create((typeof Iterator == "function" ? Iterator : Object).prototype);
    return s.next = a(0), s.throw = a(1), s.return = a(2), typeof Symbol == "function" && (s[Symbol.iterator] = function() {
        return this;
    }), s;
    "TURBOPACK unreachable";
    function a(u) {
        return function(l) {
            return c([
                u,
                l
            ]);
        };
    }
    function c(u) {
        if (r) throw new TypeError("Generator is already executing.");
        for(; s && (s = 0, u[0] && (n = 0)), n;)try {
            if (r = 1, i && (o = u[0] & 2 ? i.return : u[0] ? i.throw || ((o = i.return) && o.call(i), 0) : i.next) && !(o = o.call(i, u[1])).done) return o;
            switch(i = 0, o && (u = [
                u[0] & 2,
                o.value
            ]), u[0]){
                case 0:
                case 1:
                    o = u;
                    break;
                case 4:
                    return n.label++, {
                        value: u[1],
                        done: !1
                    };
                case 5:
                    n.label++, i = u[1], u = [
                        0
                    ];
                    continue;
                case 7:
                    u = n.ops.pop(), n.trys.pop();
                    continue;
                default:
                    if (o = n.trys, !(o = o.length > 0 && o[o.length - 1]) && (u[0] === 6 || u[0] === 2)) {
                        n = 0;
                        continue;
                    }
                    if (u[0] === 3 && (!o || u[1] > o[0] && u[1] < o[3])) {
                        n.label = u[1];
                        break;
                    }
                    if (u[0] === 6 && n.label < o[1]) {
                        n.label = o[1], o = u;
                        break;
                    }
                    if (o && n.label < o[2]) {
                        n.label = o[2], n.ops.push(u);
                        break;
                    }
                    o[2] && n.ops.pop(), n.trys.pop();
                    continue;
            }
            u = t.call(e, n);
        } catch (l) {
            u = [
                6,
                l
            ], i = 0;
        } finally{
            r = o = 0;
        }
        if (u[0] & 5) throw u[1];
        return {
            value: u[0] ? u[1] : void 0,
            done: !0
        };
    }
}
function ge(e) {
    var t = typeof Symbol == "function" && Symbol.iterator, n = t && e[t], r = 0;
    if (n) return n.call(e);
    if (e && typeof e.length == "number") return {
        next: function() {
            return e && r >= e.length && (e = void 0), {
                value: e && e[r++],
                done: !e
            };
        }
    };
    throw new TypeError(t ? "Object is not iterable." : "Symbol.iterator is not defined.");
}
function le(e, t) {
    var n = typeof Symbol == "function" && e[Symbol.iterator];
    if (!n) return e;
    var r = n.call(e), i, o = [], s;
    try {
        for(; (t === void 0 || t-- > 0) && !(i = r.next()).done;)o.push(i.value);
    } catch (a) {
        s = {
            error: a
        };
    } finally{
        try {
            i && !i.done && (n = r.return) && n.call(r);
        } finally{
            if (s) throw s.error;
        }
    }
    return o;
}
function be(e, t, n) {
    if (n || arguments.length === 2) for(var r = 0, i = t.length, o; r < i; r++)(o || !(r in t)) && (o || (o = Array.prototype.slice.call(t, 0, r)), o[r] = t[r]);
    return e.concat(o || Array.prototype.slice.call(t));
}
function he(e) {
    return this instanceof he ? (this.v = e, this) : new he(e);
}
function Mr(e, t, n) {
    if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
    var r = n.apply(e, t || []), i, o = [];
    return i = Object.create((typeof AsyncIterator == "function" ? AsyncIterator : Object).prototype), a("next"), a("throw"), a("return", s), i[Symbol.asyncIterator] = function() {
        return this;
    }, i;
    "TURBOPACK unreachable";
    function s(d) {
        return function(m) {
            return Promise.resolve(m).then(d, f);
        };
    }
    function a(d, m) {
        r[d] && (i[d] = function(p) {
            return new Promise(function(g, h) {
                o.push([
                    d,
                    p,
                    g,
                    h
                ]) > 1 || c(d, p);
            });
        }, m && (i[d] = m(i[d])));
    }
    function c(d, m) {
        try {
            u(r[d](m));
        } catch (p) {
            v(o[0][3], p);
        }
    }
    function u(d) {
        d.value instanceof he ? Promise.resolve(d.value.v).then(l, f) : v(o[0][2], d);
    }
    function l(d) {
        c("next", d);
    }
    function f(d) {
        c("throw", d);
    }
    function v(d, m) {
        d(m), o.shift(), o.length && c(o[0][0], o[0][1]);
    }
}
function Rr(e) {
    if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
    var t = e[Symbol.asyncIterator], n;
    return t ? t.call(e) : (e = typeof ge == "function" ? ge(e) : e[Symbol.iterator](), n = {}, r("next"), r("throw"), r("return"), n[Symbol.asyncIterator] = function() {
        return this;
    }, n);
    "TURBOPACK unreachable";
    function r(o) {
        n[o] = e[o] && function(s) {
            return new Promise(function(a, c) {
                s = e[o](s), i(a, c, s.done, s.value);
            });
        };
    }
    function i(o, s, a, c) {
        Promise.resolve(c).then(function(u) {
            o({
                value: u,
                done: a
            });
        }, s);
    }
}
function A(e) {
    return typeof e == "function";
}
function vt(e) {
    var t = function(r) {
        Error.call(r), r.stack = new Error().stack;
    }, n = e(t);
    return n.prototype = Object.create(Error.prototype), n.prototype.constructor = n, n;
}
var Ze = vt(function(e) {
    return function(n) {
        e(this), this.message = n ? n.length + ` errors occurred during unsubscription:
` + n.map(function(r, i) {
            return i + 1 + ") " + r.toString();
        }).join(`
  `) : "", this.name = "UnsubscriptionError", this.errors = n;
    };
});
function Ue(e, t) {
    if (e) {
        var n = e.indexOf(t);
        0 <= n && e.splice(n, 1);
    }
}
var Oe = function() {
    function e(t) {
        this.initialTeardown = t, this.closed = !1, this._parentage = null, this._finalizers = null;
    }
    return e.prototype.unsubscribe = function() {
        var t, n, r, i, o;
        if (!this.closed) {
            this.closed = !0;
            var s = this._parentage;
            if (s) if (this._parentage = null, Array.isArray(s)) try {
                for(var a = ge(s), c = a.next(); !c.done; c = a.next()){
                    var u = c.value;
                    u.remove(this);
                }
            } catch (p) {
                t = {
                    error: p
                };
            } finally{
                try {
                    c && !c.done && (n = a.return) && n.call(a);
                } finally{
                    if (t) throw t.error;
                }
            }
            else s.remove(this);
            var l = this.initialTeardown;
            if (A(l)) try {
                l();
            } catch (p) {
                o = p instanceof Ze ? p.errors : [
                    p
                ];
            }
            var f = this._finalizers;
            if (f) {
                this._finalizers = null;
                try {
                    for(var v = ge(f), d = v.next(); !d.done; d = v.next()){
                        var m = d.value;
                        try {
                            Lt(m);
                        } catch (p) {
                            o = o ?? [], p instanceof Ze ? o = be(be([], le(o)), le(p.errors)) : o.push(p);
                        }
                    }
                } catch (p) {
                    r = {
                        error: p
                    };
                } finally{
                    try {
                        d && !d.done && (i = v.return) && i.call(v);
                    } finally{
                        if (r) throw r.error;
                    }
                }
            }
            if (o) throw new Ze(o);
        }
    }, e.prototype.add = function(t) {
        var n;
        if (t && t !== this) if (this.closed) Lt(t);
        else {
            if (t instanceof e) {
                if (t.closed || t._hasParent(this)) return;
                t._addParent(this);
            }
            (this._finalizers = (n = this._finalizers) !== null && n !== void 0 ? n : []).push(t);
        }
    }, e.prototype._hasParent = function(t) {
        var n = this._parentage;
        return n === t || Array.isArray(n) && n.includes(t);
    }, e.prototype._addParent = function(t) {
        var n = this._parentage;
        this._parentage = Array.isArray(n) ? (n.push(t), n) : n ? [
            n,
            t
        ] : t;
    }, e.prototype._removeParent = function(t) {
        var n = this._parentage;
        n === t ? this._parentage = null : Array.isArray(n) && Ue(n, t);
    }, e.prototype.remove = function(t) {
        var n = this._finalizers;
        n && Ue(n, t), t instanceof e && t._removeParent(this);
    }, e.EMPTY = function() {
        var t = new e();
        return t.closed = !0, t;
    }(), e;
}(), an = Oe.EMPTY;
function cn(e) {
    return e instanceof Oe || e && "closed" in e && A(e.remove) && A(e.add) && A(e.unsubscribe);
}
function Lt(e) {
    A(e) ? e() : e.unsubscribe();
}
var Dr = {
    Promise: void 0
}, $r = {
    setTimeout: function(e, t) {
        for(var n = [], r = 2; r < arguments.length; r++)n[r - 2] = arguments[r];
        return setTimeout.apply(void 0, be([
            e,
            t
        ], le(n)));
    },
    clearTimeout: function(e) {
        return clearTimeout(e);
    },
    delegate: void 0
};
function un(e) {
    $r.setTimeout(function() {
        throw e;
    });
}
function je() {}
function Re(e) {
    e();
}
var mt = function(e) {
    te(t, e);
    function t(n) {
        var r = e.call(this) || this;
        return r.isStopped = !1, n ? (r.destination = n, cn(n) && n.add(r)) : r.destination = Ur, r;
    }
    return t.create = function(n, r, i) {
        return new Ce(n, r, i);
    }, t.prototype.next = function(n) {
        this.isStopped || this._next(n);
    }, t.prototype.error = function(n) {
        this.isStopped || (this.isStopped = !0, this._error(n));
    }, t.prototype.complete = function() {
        this.isStopped || (this.isStopped = !0, this._complete());
    }, t.prototype.unsubscribe = function() {
        this.closed || (this.isStopped = !0, e.prototype.unsubscribe.call(this), this.destination = null);
    }, t.prototype._next = function(n) {
        this.destination.next(n);
    }, t.prototype._error = function(n) {
        try {
            this.destination.error(n);
        } finally{
            this.unsubscribe();
        }
    }, t.prototype._complete = function() {
        try {
            this.destination.complete();
        } finally{
            this.unsubscribe();
        }
    }, t;
}(Oe), Nr = function() {
    function e(t) {
        this.partialObserver = t;
    }
    return e.prototype.next = function(t) {
        var n = this.partialObserver;
        if (n.next) try {
            n.next(t);
        } catch (r) {
            _e(r);
        }
    }, e.prototype.error = function(t) {
        var n = this.partialObserver;
        if (n.error) try {
            n.error(t);
        } catch (r) {
            _e(r);
        }
        else _e(t);
    }, e.prototype.complete = function() {
        var t = this.partialObserver;
        if (t.complete) try {
            t.complete();
        } catch (n) {
            _e(n);
        }
    }, e;
}(), Ce = function(e) {
    te(t, e);
    function t(n, r, i) {
        var o = e.call(this) || this, s;
        return A(n) || !n ? s = {
            next: n ?? void 0,
            error: r ?? void 0,
            complete: i ?? void 0
        } : s = n, o.destination = new Nr(s), o;
    }
    return t;
}(mt);
function _e(e) {
    un(e);
}
function Fr(e) {
    throw e;
}
var Ur = {
    closed: !0,
    next: je,
    error: Fr,
    complete: je
}, gt = function() {
    return typeof Symbol == "function" && Symbol.observable || "@@observable";
}();
function bt(e) {
    return e;
}
function jr(e) {
    return e.length === 0 ? bt : e.length === 1 ? e[0] : function(n) {
        return e.reduce(function(r, i) {
            return i(r);
        }, n);
    };
}
var k = function() {
    function e(t) {
        t && (this._subscribe = t);
    }
    return e.prototype.lift = function(t) {
        var n = new e();
        return n.source = this, n.operator = t, n;
    }, e.prototype.subscribe = function(t, n, r) {
        var i = this, o = Br(t) ? t : new Ce(t, n, r);
        return Re(function() {
            var s = i, a = s.operator, c = s.source;
            o.add(a ? a.call(o, c) : c ? i._subscribe(o) : i._trySubscribe(o));
        }), o;
    }, e.prototype._trySubscribe = function(t) {
        try {
            return this._subscribe(t);
        } catch (n) {
            t.error(n);
        }
    }, e.prototype.forEach = function(t, n) {
        var r = this;
        return n = _t(n), new n(function(i, o) {
            var s = new Ce({
                next: function(a) {
                    try {
                        t(a);
                    } catch (c) {
                        o(c), s.unsubscribe();
                    }
                },
                error: o,
                complete: i
            });
            r.subscribe(s);
        });
    }, e.prototype._subscribe = function(t) {
        var n;
        return (n = this.source) === null || n === void 0 ? void 0 : n.subscribe(t);
    }, e.prototype[gt] = function() {
        return this;
    }, e.prototype.pipe = function() {
        for(var t = [], n = 0; n < arguments.length; n++)t[n] = arguments[n];
        return jr(t)(this);
    }, e.prototype.toPromise = function(t) {
        var n = this;
        return t = _t(t), new t(function(r, i) {
            var o;
            n.subscribe(function(s) {
                return o = s;
            }, function(s) {
                return i(s);
            }, function() {
                return r(o);
            });
        });
    }, e.create = function(t) {
        return new e(t);
    }, e;
}();
function _t(e) {
    var t;
    return (t = e ?? Dr.Promise) !== null && t !== void 0 ? t : Promise;
}
function Wr(e) {
    return e && A(e.next) && A(e.error) && A(e.complete);
}
function Br(e) {
    return e && e instanceof mt || Wr(e) && cn(e);
}
function Vr(e) {
    return A(e == null ? void 0 : e.lift);
}
function B(e) {
    return function(t) {
        if (Vr(t)) return t.lift(function(n) {
            try {
                return e(n, this);
            } catch (r) {
                this.error(r);
            }
        });
        throw new TypeError("Unable to lift unknown Observable type");
    };
}
function W(e, t, n, r, i) {
    return new Hr(e, t, n, r, i);
}
var Hr = function(e) {
    te(t, e);
    function t(n, r, i, o, s, a) {
        var c = e.call(this, n) || this;
        return c.onFinalize = s, c.shouldUnsubscribe = a, c._next = r ? function(u) {
            try {
                r(u);
            } catch (l) {
                n.error(l);
            }
        } : e.prototype._next, c._error = o ? function(u) {
            try {
                o(u);
            } catch (l) {
                n.error(l);
            } finally{
                this.unsubscribe();
            }
        } : e.prototype._error, c._complete = i ? function() {
            try {
                i();
            } catch (u) {
                n.error(u);
            } finally{
                this.unsubscribe();
            }
        } : e.prototype._complete, c;
    }
    return t.prototype.unsubscribe = function() {
        var n;
        if (!this.shouldUnsubscribe || this.shouldUnsubscribe()) {
            var r = this.closed;
            e.prototype.unsubscribe.call(this), !r && ((n = this.onFinalize) === null || n === void 0 || n.call(this));
        }
    }, t;
}(mt), zr = vt(function(e) {
    return function() {
        e(this), this.name = "ObjectUnsubscribedError", this.message = "object unsubscribed";
    };
}), ee = function(e) {
    te(t, e);
    function t() {
        var n = e.call(this) || this;
        return n.closed = !1, n.currentObservers = null, n.observers = [], n.isStopped = !1, n.hasError = !1, n.thrownError = null, n;
    }
    return t.prototype.lift = function(n) {
        var r = new It(this, this);
        return r.operator = n, r;
    }, t.prototype._throwIfClosed = function() {
        if (this.closed) throw new zr();
    }, t.prototype.next = function(n) {
        var r = this;
        Re(function() {
            var i, o;
            if (r._throwIfClosed(), !r.isStopped) {
                r.currentObservers || (r.currentObservers = Array.from(r.observers));
                try {
                    for(var s = ge(r.currentObservers), a = s.next(); !a.done; a = s.next()){
                        var c = a.value;
                        c.next(n);
                    }
                } catch (u) {
                    i = {
                        error: u
                    };
                } finally{
                    try {
                        a && !a.done && (o = s.return) && o.call(s);
                    } finally{
                        if (i) throw i.error;
                    }
                }
            }
        });
    }, t.prototype.error = function(n) {
        var r = this;
        Re(function() {
            if (r._throwIfClosed(), !r.isStopped) {
                r.hasError = r.isStopped = !0, r.thrownError = n;
                for(var i = r.observers; i.length;)i.shift().error(n);
            }
        });
    }, t.prototype.complete = function() {
        var n = this;
        Re(function() {
            if (n._throwIfClosed(), !n.isStopped) {
                n.isStopped = !0;
                for(var r = n.observers; r.length;)r.shift().complete();
            }
        });
    }, t.prototype.unsubscribe = function() {
        this.isStopped = this.closed = !0, this.observers = this.currentObservers = null;
    }, Object.defineProperty(t.prototype, "observed", {
        get: function() {
            var n;
            return ((n = this.observers) === null || n === void 0 ? void 0 : n.length) > 0;
        },
        enumerable: !1,
        configurable: !0
    }), t.prototype._trySubscribe = function(n) {
        return this._throwIfClosed(), e.prototype._trySubscribe.call(this, n);
    }, t.prototype._subscribe = function(n) {
        return this._throwIfClosed(), this._checkFinalizedStatuses(n), this._innerSubscribe(n);
    }, t.prototype._innerSubscribe = function(n) {
        var r = this, i = this, o = i.hasError, s = i.isStopped, a = i.observers;
        return o || s ? an : (this.currentObservers = null, a.push(n), new Oe(function() {
            r.currentObservers = null, Ue(a, n);
        }));
    }, t.prototype._checkFinalizedStatuses = function(n) {
        var r = this, i = r.hasError, o = r.thrownError, s = r.isStopped;
        i ? n.error(o) : s && n.complete();
    }, t.prototype.asObservable = function() {
        var n = new k();
        return n.source = this, n;
    }, t.create = function(n, r) {
        return new It(n, r);
    }, t;
}(k), It = function(e) {
    te(t, e);
    function t(n, r) {
        var i = e.call(this) || this;
        return i.destination = n, i.source = r, i;
    }
    return t.prototype.next = function(n) {
        var r, i;
        (i = (r = this.destination) === null || r === void 0 ? void 0 : r.next) === null || i === void 0 || i.call(r, n);
    }, t.prototype.error = function(n) {
        var r, i;
        (i = (r = this.destination) === null || r === void 0 ? void 0 : r.error) === null || i === void 0 || i.call(r, n);
    }, t.prototype.complete = function() {
        var n, r;
        (r = (n = this.destination) === null || n === void 0 ? void 0 : n.complete) === null || r === void 0 || r.call(n);
    }, t.prototype._subscribe = function(n) {
        var r, i;
        return (i = (r = this.source) === null || r === void 0 ? void 0 : r.subscribe(n)) !== null && i !== void 0 ? i : an;
    }, t;
}(ee), ln = function(e) {
    te(t, e);
    function t(n) {
        var r = e.call(this) || this;
        return r._value = n, r;
    }
    return Object.defineProperty(t.prototype, "value", {
        get: function() {
            return this.getValue();
        },
        enumerable: !1,
        configurable: !0
    }), t.prototype._subscribe = function(n) {
        var r = e.prototype._subscribe.call(this, n);
        return !r.closed && n.next(this._value), r;
    }, t.prototype.getValue = function() {
        var n = this, r = n.hasError, i = n.thrownError, o = n._value;
        if (r) throw i;
        return this._throwIfClosed(), o;
    }, t.prototype.next = function(n) {
        e.prototype.next.call(this, this._value = n);
    }, t;
}(ee), Yr = {
    now: function() {
        return Date.now();
    }
}, qr = function(e) {
    te(t, e);
    function t(n, r) {
        return e.call(this) || this;
    }
    return t.prototype.schedule = function(n, r) {
        return this;
    }, t;
}(Oe), Mt = {
    setInterval: function(e, t) {
        for(var n = [], r = 2; r < arguments.length; r++)n[r - 2] = arguments[r];
        return setInterval.apply(void 0, be([
            e,
            t
        ], le(n)));
    },
    clearInterval: function(e) {
        return clearInterval(e);
    },
    delegate: void 0
}, Kr = function(e) {
    te(t, e);
    function t(n, r) {
        var i = e.call(this, n, r) || this;
        return i.scheduler = n, i.work = r, i.pending = !1, i;
    }
    return t.prototype.schedule = function(n, r) {
        var i;
        if (r === void 0 && (r = 0), this.closed) return this;
        this.state = n;
        var o = this.id, s = this.scheduler;
        return o != null && (this.id = this.recycleAsyncId(s, o, r)), this.pending = !0, this.delay = r, this.id = (i = this.id) !== null && i !== void 0 ? i : this.requestAsyncId(s, this.id, r), this;
    }, t.prototype.requestAsyncId = function(n, r, i) {
        return i === void 0 && (i = 0), Mt.setInterval(n.flush.bind(n, this), i);
    }, t.prototype.recycleAsyncId = function(n, r, i) {
        if (i === void 0 && (i = 0), i != null && this.delay === i && this.pending === !1) return r;
        r != null && Mt.clearInterval(r);
    }, t.prototype.execute = function(n, r) {
        if (this.closed) return new Error("executing a cancelled action");
        this.pending = !1;
        var i = this._execute(n, r);
        if (i) return i;
        this.pending === !1 && this.id != null && (this.id = this.recycleAsyncId(this.scheduler, this.id, null));
    }, t.prototype._execute = function(n, r) {
        var i = !1, o;
        try {
            this.work(n);
        } catch (s) {
            i = !0, o = s || new Error("Scheduled action threw falsy error");
        }
        if (i) return this.unsubscribe(), o;
    }, t.prototype.unsubscribe = function() {
        if (!this.closed) {
            var n = this, r = n.id, i = n.scheduler, o = i.actions;
            this.work = this.state = this.scheduler = null, this.pending = !1, Ue(o, this), r != null && (this.id = this.recycleAsyncId(i, r, null)), this.delay = null, e.prototype.unsubscribe.call(this);
        }
    }, t;
}(qr), Rt = function() {
    function e(t, n) {
        n === void 0 && (n = e.now), this.schedulerActionCtor = t, this.now = n;
    }
    return e.prototype.schedule = function(t, n, r) {
        return n === void 0 && (n = 0), new this.schedulerActionCtor(this, t).schedule(r, n);
    }, e.now = Yr.now, e;
}(), Gr = function(e) {
    te(t, e);
    function t(n, r) {
        r === void 0 && (r = Rt.now);
        var i = e.call(this, n, r) || this;
        return i.actions = [], i._active = !1, i;
    }
    return t.prototype.flush = function(n) {
        var r = this.actions;
        if (this._active) {
            r.push(n);
            return;
        }
        var i;
        this._active = !0;
        do if (i = n.execute(n.state, n.delay)) break;
        while (n = r.shift())
        if (this._active = !1, i) {
            for(; n = r.shift();)n.unsubscribe();
            throw i;
        }
    }, t;
}(Rt), Qr = new Gr(Kr);
function Jr(e) {
    return e && A(e.schedule);
}
function Xr(e) {
    return e[e.length - 1];
}
function yt(e) {
    return Jr(Xr(e)) ? e.pop() : void 0;
}
var wt = function(e) {
    return e && typeof e.length == "number" && typeof e != "function";
};
function fn(e) {
    return A(e == null ? void 0 : e.then);
}
function dn(e) {
    return A(e[gt]);
}
function pn(e) {
    return Symbol.asyncIterator && A(e == null ? void 0 : e[Symbol.asyncIterator]);
}
function hn(e) {
    return new TypeError("You provided " + (e !== null && typeof e == "object" ? "an invalid object" : "'" + e + "'") + " where a stream was expected. You can provide an Observable, Promise, ReadableStream, Array, AsyncIterable, or Iterable.");
}
function Zr() {
    return typeof Symbol != "function" || !Symbol.iterator ? "@@iterator" : Symbol.iterator;
}
var vn = Zr();
function mn(e) {
    return A(e == null ? void 0 : e[vn]);
}
function gn(e) {
    return Mr(this, arguments, function() {
        var n, r, i, o;
        return sn(this, function(s) {
            switch(s.label){
                case 0:
                    n = e.getReader(), s.label = 1;
                case 1:
                    s.trys.push([
                        1,
                        ,
                        9,
                        10
                    ]), s.label = 2;
                case 2:
                    return [
                        4,
                        he(n.read())
                    ];
                case 3:
                    return r = s.sent(), i = r.value, o = r.done, o ? [
                        4,
                        he(void 0)
                    ] : [
                        3,
                        5
                    ];
                case 4:
                    return [
                        2,
                        s.sent()
                    ];
                case 5:
                    return [
                        4,
                        he(i)
                    ];
                case 6:
                    return [
                        4,
                        s.sent()
                    ];
                case 7:
                    return s.sent(), [
                        3,
                        2
                    ];
                case 8:
                    return [
                        3,
                        10
                    ];
                case 9:
                    return n.releaseLock(), [
                        7
                    ];
                case 10:
                    return [
                        2
                    ];
            }
        });
    });
}
function bn(e) {
    return A(e == null ? void 0 : e.getReader);
}
function H(e) {
    if (e instanceof k) return e;
    if (e != null) {
        if (dn(e)) return ei(e);
        if (wt(e)) return ti(e);
        if (fn(e)) return ni(e);
        if (pn(e)) return yn(e);
        if (mn(e)) return ri(e);
        if (bn(e)) return ii(e);
    }
    throw hn(e);
}
function ei(e) {
    return new k(function(t) {
        var n = e[gt]();
        if (A(n.subscribe)) return n.subscribe(t);
        throw new TypeError("Provided object does not correctly implement Symbol.observable");
    });
}
function ti(e) {
    return new k(function(t) {
        for(var n = 0; n < e.length && !t.closed; n++)t.next(e[n]);
        t.complete();
    });
}
function ni(e) {
    return new k(function(t) {
        e.then(function(n) {
            t.closed || (t.next(n), t.complete());
        }, function(n) {
            return t.error(n);
        }).then(null, un);
    });
}
function ri(e) {
    return new k(function(t) {
        var n, r;
        try {
            for(var i = ge(e), o = i.next(); !o.done; o = i.next()){
                var s = o.value;
                if (t.next(s), t.closed) return;
            }
        } catch (a) {
            n = {
                error: a
            };
        } finally{
            try {
                o && !o.done && (r = i.return) && r.call(i);
            } finally{
                if (n) throw n.error;
            }
        }
        t.complete();
    });
}
function yn(e) {
    return new k(function(t) {
        oi(e, t).catch(function(n) {
            return t.error(n);
        });
    });
}
function ii(e) {
    return yn(gn(e));
}
function oi(e, t) {
    var n, r, i, o;
    return Ir(this, void 0, void 0, function() {
        var s, a;
        return sn(this, function(c) {
            switch(c.label){
                case 0:
                    c.trys.push([
                        0,
                        5,
                        6,
                        11
                    ]), n = Rr(e), c.label = 1;
                case 1:
                    return [
                        4,
                        n.next()
                    ];
                case 2:
                    if (r = c.sent(), !!r.done) return [
                        3,
                        4
                    ];
                    if (s = r.value, t.next(s), t.closed) return [
                        2
                    ];
                    c.label = 3;
                case 3:
                    return [
                        3,
                        1
                    ];
                case 4:
                    return [
                        3,
                        11
                    ];
                case 5:
                    return a = c.sent(), i = {
                        error: a
                    }, [
                        3,
                        11
                    ];
                case 6:
                    return c.trys.push([
                        6,
                        ,
                        9,
                        10
                    ]), r && !r.done && (o = n.return) ? [
                        4,
                        o.call(n)
                    ] : [
                        3,
                        8
                    ];
                case 7:
                    c.sent(), c.label = 8;
                case 8:
                    return [
                        3,
                        10
                    ];
                case 9:
                    if (i) throw i.error;
                    return [
                        7
                    ];
                case 10:
                    return [
                        7
                    ];
                case 11:
                    return t.complete(), [
                        2
                    ];
            }
        });
    });
}
function ie(e, t, n, r, i) {
    r === void 0 && (r = 0), i === void 0 && (i = !1);
    var o = t.schedule(function() {
        n(), i ? e.add(this.schedule(null, r)) : this.unsubscribe();
    }, r);
    if (e.add(o), !i) return o;
}
function wn(e, t) {
    return t === void 0 && (t = 0), B(function(n, r) {
        n.subscribe(W(r, function(i) {
            return ie(r, e, function() {
                return r.next(i);
            }, t);
        }, function() {
            return ie(r, e, function() {
                return r.complete();
            }, t);
        }, function(i) {
            return ie(r, e, function() {
                return r.error(i);
            }, t);
        }));
    });
}
function xn(e, t) {
    return t === void 0 && (t = 0), B(function(n, r) {
        r.add(e.schedule(function() {
            return n.subscribe(r);
        }, t));
    });
}
function si(e, t) {
    return H(e).pipe(xn(t), wn(t));
}
function ai(e, t) {
    return H(e).pipe(xn(t), wn(t));
}
function ci(e, t) {
    return new k(function(n) {
        var r = 0;
        return t.schedule(function() {
            r === e.length ? n.complete() : (n.next(e[r++]), n.closed || this.schedule());
        });
    });
}
function ui(e, t) {
    return new k(function(n) {
        var r;
        return ie(n, t, function() {
            r = e[vn](), ie(n, t, function() {
                var i, o, s;
                try {
                    i = r.next(), o = i.value, s = i.done;
                } catch (a) {
                    n.error(a);
                    return;
                }
                s ? n.complete() : n.next(o);
            }, 0, !0);
        }), function() {
            return A(r == null ? void 0 : r.return) && r.return();
        };
    });
}
function Sn(e, t) {
    if (!e) throw new Error("Iterable cannot be null");
    return new k(function(n) {
        ie(n, t, function() {
            var r = e[Symbol.asyncIterator]();
            ie(n, t, function() {
                r.next().then(function(i) {
                    i.done ? n.complete() : n.next(i.value);
                });
            }, 0, !0);
        });
    });
}
function li(e, t) {
    return Sn(gn(e), t);
}
function fi(e, t) {
    if (e != null) {
        if (dn(e)) return si(e, t);
        if (wt(e)) return ci(e, t);
        if (fn(e)) return ai(e, t);
        if (pn(e)) return Sn(e, t);
        if (mn(e)) return ui(e, t);
        if (bn(e)) return li(e, t);
    }
    throw hn(e);
}
function Ge(e, t) {
    return t ? fi(e, t) : H(e);
}
function Dt() {
    for(var e = [], t = 0; t < arguments.length; t++)e[t] = arguments[t];
    var n = yt(e);
    return Ge(e, n);
}
function di(e) {
    return e instanceof Date && !isNaN(e);
}
var pi = vt(function(e) {
    return function(n) {
        n === void 0 && (n = null), e(this), this.message = "Timeout has occurred", this.name = "TimeoutError", this.info = n;
    };
});
function hi(e, t) {
    var n = di(e) ? {
        first: e
    } : typeof e == "number" ? {
        each: e
    } : e, r = n.first, i = n.each, o = n.with, s = o === void 0 ? vi : o, a = n.scheduler, c = a === void 0 ? Qr : a, u = n.meta, l = u === void 0 ? null : u;
    if (r == null && i == null) throw new TypeError("No timeout provided.");
    return B(function(f, v) {
        var d, m, p = null, g = 0, h = function(x) {
            m = ie(v, c, function() {
                try {
                    d.unsubscribe(), H(s({
                        meta: l,
                        lastValue: p,
                        seen: g
                    })).subscribe(v);
                } catch (E) {
                    v.error(E);
                }
            }, x);
        };
        d = f.subscribe(W(v, function(x) {
            m == null || m.unsubscribe(), g++, v.next(p = x), i > 0 && h(i);
        }, void 0, void 0, function() {
            m != null && m.closed || m == null || m.unsubscribe(), p = null;
        })), !g && h(r != null ? typeof r == "number" ? r : +r - c.now() : i);
    });
}
function vi(e) {
    throw new pi(e);
}
function O(e, t) {
    return B(function(n, r) {
        var i = 0;
        n.subscribe(W(r, function(o) {
            r.next(e.call(t, o, i++));
        }));
    });
}
var mi = Array.isArray;
function gi(e, t) {
    return mi(t) ? e.apply(void 0, be([], le(t))) : e(t);
}
function bi(e) {
    return O(function(t) {
        return gi(e, t);
    });
}
function yi(e, t, n, r, i, o, s, a) {
    var c = [], u = 0, l = 0, f = !1, v = function() {
        f && !c.length && !u && t.complete();
    }, d = function(p) {
        return u < r ? m(p) : c.push(p);
    }, m = function(p) {
        u++;
        var g = !1;
        H(n(p, l++)).subscribe(W(t, function(h) {
            t.next(h);
        }, function() {
            g = !0;
        }, void 0, function() {
            if (g) try {
                u--;
                for(var h = function() {
                    var x = c.shift();
                    s || m(x);
                }; c.length && u < r;)h();
                v();
            } catch (x) {
                t.error(x);
            }
        }));
    };
    return e.subscribe(W(t, d, function() {
        f = !0, v();
    })), function() {};
}
function xt(e, t, n) {
    return n === void 0 && (n = 1 / 0), A(t) ? xt(function(r, i) {
        return O(function(o, s) {
            return t(r, o, i, s);
        })(H(e(r, i)));
    }, n) : (typeof t == "number" && (n = t), B(function(r, i) {
        return yi(r, i, e, n);
    }));
}
function wi(e) {
    return xt(bt, e);
}
function xi() {
    return wi(1);
}
function We() {
    for(var e = [], t = 0; t < arguments.length; t++)e[t] = arguments[t];
    return xi()(Ge(e, yt(e)));
}
var Si = [
    "addListener",
    "removeListener"
], Ti = [
    "addEventListener",
    "removeEventListener"
], Ei = [
    "on",
    "off"
];
function at(e, t, n, r) {
    if (A(n) && (r = n, n = void 0), r) return at(e, t, n).pipe(bi(r));
    var i = le(Ai(e) ? Ti.map(function(a) {
        return function(c) {
            return e[a](t, c, n);
        };
    }) : Ci(e) ? Si.map($t(e, t)) : Pi(e) ? Ei.map($t(e, t)) : [], 2), o = i[0], s = i[1];
    if (!o && wt(e)) return xt(function(a) {
        return at(a, t, n);
    })(H(e));
    if (!o) throw new TypeError("Invalid event target");
    return new k(function(a) {
        var c = function() {
            for(var u = [], l = 0; l < arguments.length; l++)u[l] = arguments[l];
            return a.next(1 < u.length ? u : u[0]);
        };
        return o(c), function() {
            return s(c);
        };
    });
}
function $t(e, t) {
    return function(n) {
        return function(r) {
            return e[n](t, r);
        };
    };
}
function Ci(e) {
    return A(e.addListener) && A(e.removeListener);
}
function Pi(e) {
    return A(e.on) && A(e.off);
}
function Ai(e) {
    return A(e.addEventListener) && A(e.removeEventListener);
}
function Qe(e, t) {
    return B(function(n, r) {
        var i = 0;
        n.subscribe(W(r, function(o) {
            return e.call(t, o, i++) && r.next(o);
        }));
    });
}
function Oi(e, t, n, r, i) {
    return function(o, s) {
        var a = n, c = t, u = 0;
        o.subscribe(W(s, function(l) {
            var f = u++;
            c = a ? e(c, l, f) : (a = !0, l), s.next(c);
        }, i));
    };
}
function ki(e, t) {
    return t === void 0 && (t = bt), e = e ?? Li, B(function(n, r) {
        var i, o = !0;
        n.subscribe(W(r, function(s) {
            var a = t(s);
            (o || !e(i, a)) && (o = !1, i = a, r.next(s));
        }));
    });
}
function Li(e, t) {
    return e === t;
}
function ct(e, t) {
    return B(Oi(e, t, arguments.length >= 2, !0));
}
function _i(e) {
    e === void 0 && (e = {});
    var t = e.connector, n = t === void 0 ? function() {
        return new ee();
    } : t, r = e.resetOnError, i = r === void 0 ? !0 : r, o = e.resetOnComplete, s = o === void 0 ? !0 : o, a = e.resetOnRefCountZero, c = a === void 0 ? !0 : a;
    return function(u) {
        var l, f, v, d = 0, m = !1, p = !1, g = function() {
            f == null || f.unsubscribe(), f = void 0;
        }, h = function() {
            g(), l = v = void 0, m = p = !1;
        }, x = function() {
            var E = l;
            h(), E == null || E.unsubscribe();
        };
        return B(function(E, P) {
            d++, !p && !m && g();
            var b = v = v ?? n();
            P.add(function() {
                d--, d === 0 && !p && !m && (f = et(x, c));
            }), b.subscribe(P), !l && d > 0 && (l = new Ce({
                next: function(S) {
                    return b.next(S);
                },
                error: function(S) {
                    p = !0, g(), f = et(h, i, S), b.error(S);
                },
                complete: function() {
                    m = !0, g(), f = et(h, s), b.complete();
                }
            }), H(E).subscribe(l));
        })(u);
    };
}
function et(e, t) {
    for(var n = [], r = 2; r < arguments.length; r++)n[r - 2] = arguments[r];
    if (t === !0) {
        e();
        return;
    }
    if (t !== !1) {
        var i = new Ce({
            next: function() {
                i.unsubscribe(), e();
            }
        });
        return H(t.apply(void 0, be([], le(n)))).subscribe(i);
    }
}
function Ii(e) {
    return B(function(t, n) {
        var r = !1, i = W(n, function() {
            i == null || i.unsubscribe(), r = !0;
        }, je);
        H(e).subscribe(i), t.subscribe(W(n, function(o) {
            return r && n.next(o);
        }));
    });
}
function D() {
    for(var e = [], t = 0; t < arguments.length; t++)e[t] = arguments[t];
    var n = yt(e);
    return B(function(r, i) {
        (n ? We(e, r, n) : We(e, r)).subscribe(i);
    });
}
function Tn(e, t) {
    return B(function(n, r) {
        var i = null, o = 0, s = !1, a = function() {
            return s && !i && r.complete();
        };
        n.subscribe(W(r, function(c) {
            i == null || i.unsubscribe();
            var u = 0, l = o++;
            H(e(c, l)).subscribe(i = W(r, function(f) {
                return r.next(t ? t(c, f, l, u++) : f);
            }, function() {
                i = null, a();
            }));
        }, function() {
            s = !0, a();
        }));
    });
}
function Nt(e) {
    return B(function(t, n) {
        H(e).subscribe(W(n, function() {
            return n.complete();
        }, je)), !n.closed && t.subscribe(n);
    });
}
var Mi = Object.defineProperty, Ri = Object.defineProperties, Di = Object.getOwnPropertyDescriptors, Ft = Object.getOwnPropertySymbols, $i = Object.prototype.hasOwnProperty, Ni = Object.prototype.propertyIsEnumerable, Ut = (e, t, n)=>t in e ? Mi(e, t, {
        enumerable: !0,
        configurable: !0,
        writable: !0,
        value: n
    }) : e[t] = n, Y = (e, t)=>{
    for(var n in t || (t = {}))$i.call(t, n) && Ut(e, n, t[n]);
    if (Ft) for (var n of Ft(t))Ni.call(t, n) && Ut(e, n, t[n]);
    return e;
}, se = (e, t)=>Ri(e, Di(t)), V = (e, t, n)=>new Promise((r, i)=>{
        var o = (c)=>{
            try {
                a(n.next(c));
            } catch (u) {
                i(u);
            }
        }, s = (c)=>{
            try {
                a(n.throw(c));
            } catch (u) {
                i(u);
            }
        }, a = (c)=>c.done ? r(c.value) : Promise.resolve(c.value).then(o, s);
        a((n = n.apply(e, t)).next());
    }), En = "lk";
function j(e) {
    return typeof e > "u" ? !1 : Fi(e) || Ui(e);
}
function Fi(e) {
    var t;
    return e ? e.hasOwnProperty("participant") && e.hasOwnProperty("source") && e.hasOwnProperty("track") && typeof ((t = e.publication) == null ? void 0 : t.track) < "u" : !1;
}
function Ui(e) {
    return e ? e.hasOwnProperty("participant") && e.hasOwnProperty("source") && e.hasOwnProperty("publication") && typeof e.publication < "u" : !1;
}
function Pe(e) {
    return e ? e.hasOwnProperty("participant") && e.hasOwnProperty("source") && typeof e.publication > "u" : !1;
}
function N(e) {
    if (typeof e == "string" || typeof e == "number") return `${e}`;
    if (Pe(e)) return `${e.participant.identity}_${e.source}_placeholder`;
    if (j(e)) return `${e.participant.identity}_${e.publication.source}_${e.publication.trackSid}`;
    throw new Error(`Can't generate a id for the given track reference: ${e}`);
}
function Do(e, t) {
    return e === void 0 || t === void 0 ? !1 : j(e) && j(t) ? e.publication.trackSid === t.publication.trackSid : N(e) === N(t);
}
function $o(e, t) {
    return typeof t > "u" ? !1 : j(e) ? t.some((n)=>n.participant.identity === e.participant.identity && j(n) && n.publication.trackSid === e.publication.trackSid) : Pe(e) ? t.some((n)=>n.participant.identity === e.participant.identity && Pe(n) && n.source === e.source) : !1;
}
function ji(e, t) {
    return Pe(e) && j(t) && t.participant.identity === e.participant.identity && t.source === e.source;
}
function No() {
    const e = document.createElement("p");
    e.style.width = "100%", e.style.height = "200px";
    const t = document.createElement("div");
    t.style.position = "absolute", t.style.top = "0px", t.style.left = "0px", t.style.visibility = "hidden", t.style.width = "200px", t.style.height = "150px", t.style.overflow = "hidden", t.appendChild(e), document.body.appendChild(t);
    const n = e.offsetWidth;
    t.style.overflow = "scroll";
    let r = e.offsetWidth;
    return n === r && (r = t.clientWidth), document.body.removeChild(t), n - r;
}
function Fo() {
    return typeof document < "u";
}
function Wi(e) {
    e = Y({}, e);
    const t = "(?:(?:[a-z]+:)?//)?", n = "(?:\\S+(?::\\S*)?@)?", r = new RegExp("(?:25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]\\d|\\d)(?:\\.(?:25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]\\d|\\d)){3}", "g").source, u = `(?:${t}|www\\.)${n}(?:localhost|${r}|(?:(?:[a-z\\u00a1-\\uffff0-9][-_]*)*[a-z\\u00a1-\\uffff0-9]+)(?:\\.(?:[a-z\\u00a1-\\uffff0-9]-*)*[a-z\\u00a1-\\uffff0-9]+)*(?:\\.(?:[a-z\\u00a1-\\uffff]{2,}))\\.?)(?::\\d{2,5})?(?:[/?#][^\\s"]*)?`;
    return e.exact ? new RegExp(`(?:^${u}$)`, "i") : new RegExp(u, "ig");
}
var jt = "[^\\.\\s@:](?:[^\\s@:]*[^\\s@:\\.])?@[^\\.\\s@]+(?:\\.[^\\.\\s@]+)*";
function Bi({ exact: e } = {}) {
    return e ? new RegExp(`^${jt}$`) : new RegExp(jt, "g");
}
function Uo(e, t, n) {
    return Sr(e, t, ()=>V(this, null, function*() {
            const { x: i, y: o } = yield Pr(e, t, {
                placement: "top",
                middleware: [
                    Tr(6),
                    Cr(),
                    Er({
                        padding: 5
                    })
                ]
            });
            n == null || n(i, o);
        }));
}
function jo(e, t) {
    return !e.contains(t.target);
}
var Wo = ()=>({
        email: Bi(),
        url: Wi({})
    });
function Bo(e, t) {
    const n = Object.entries(t).map(([o, s], a)=>Array.from(e.matchAll(s)).map(({ index: c, 0: u })=>({
                type: o,
                weight: a,
                content: u,
                index: c ?? 0
            }))).flat().sort((o, s)=>{
        const a = o.index - s.index;
        return a !== 0 ? a : o.weight - s.weight;
    }).filter(({ index: o }, s, a)=>{
        if (s === 0) return !0;
        const c = a[s - 1];
        return c.index + c.content.length <= o;
    }), r = [];
    let i = 0;
    for (const { type: o, content: s, index: a } of n)a > i && r.push(e.substring(i, a)), r.push({
        type: o,
        content: s
    }), i = a + s.length;
    return e.length > i && r.push(e.substring(i)), r;
}
var Vi = [
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ConnectionStateChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].RoomMetadataChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ActiveSpeakersChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ConnectionQualityChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ParticipantConnected,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ParticipantDisconnected,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ParticipantPermissionsChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ParticipantMetadataChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ParticipantNameChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ParticipantAttributesChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].TrackMuted,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].TrackUnmuted,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].TrackPublished,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].TrackUnpublished,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].TrackStreamStateChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].TrackSubscriptionFailed,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].TrackSubscriptionPermissionChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].TrackSubscriptionStatusChanged
], Cn = [
    ...Vi,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].LocalTrackPublished,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].LocalTrackUnpublished
], Hi = [
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackPublished,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackUnpublished,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackMuted,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackUnmuted,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackStreamStateChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackSubscribed,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackUnsubscribed,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackSubscriptionPermissionChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackSubscriptionFailed,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].LocalTrackPublished,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].LocalTrackUnpublished
], zi = [
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].ConnectionQualityChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].IsSpeakingChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].ParticipantMetadataChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].ParticipantPermissionsChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackMuted,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackUnmuted,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackPublished,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackUnpublished,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackStreamStateChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackSubscriptionFailed,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackSubscriptionPermissionChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackSubscriptionStatusChanged
], Pn = [
    ...zi,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].LocalTrackPublished,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].LocalTrackUnpublished
], _ = _r.getLogger("lk-components-js");
_.setDefaultLevel("WARN");
function Vo(e, t = {}) {
    var n;
    _.setLevel(e), (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["setLogLevel"])((n = t.liveKitClientLogLevel) != null ? n : e);
}
function Ho(e, t = {}) {
    var n;
    const r = _.methodFactory;
    _.methodFactory = (i, o, s)=>{
        const a = r(i, o, s), c = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["LogLevel"][i], u = c >= o && c < __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["LogLevel"].silent;
        return (l, f)=>{
            f ? a(l, f) : a(l), u && e(c, l, f);
        };
    }, _.setLevel(_.getLevel()), (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["setLogExtension"])((n = t.liveKitClientLogExtension) != null ? n : e);
}
var zo = [
    {
        columns: 1,
        rows: 1
    },
    {
        columns: 1,
        rows: 2,
        orientation: "portrait"
    },
    {
        columns: 2,
        rows: 1,
        orientation: "landscape"
    },
    {
        columns: 2,
        rows: 2,
        minWidth: 560
    },
    {
        columns: 3,
        rows: 3,
        minWidth: 700
    },
    {
        columns: 4,
        rows: 4,
        minWidth: 960
    },
    {
        columns: 5,
        rows: 5,
        minWidth: 1100
    }
];
function Yi(e, t, n, r) {
    if (e.length < 1) throw new Error("At least one grid layout definition must be provided.");
    const i = qi(e);
    if (n <= 0 || r <= 0) return i[0];
    let o = 0;
    const s = n / r > 1 ? "landscape" : "portrait";
    let a = i.find((c, u, l)=>{
        o = u;
        const f = l.findIndex((v, d)=>{
            const m = !v.orientation || v.orientation === s, p = d > u, g = v.maxTiles === c.maxTiles;
            return p && g && m;
        }) !== -1;
        return c.maxTiles >= t && !f;
    });
    if (a === void 0) if (a = i[i.length - 1], a) _.warn(`No layout found for: participantCount: ${t}, width/height: ${n}/${r} fallback to biggest available layout (${a}).`);
    else throw new Error("No layout or fallback layout found.");
    if ((n < a.minWidth || r < a.minHeight) && o > 0) {
        const c = i[o - 1];
        a = Yi(i.slice(0, o), c.maxTiles, n, r);
    }
    return a;
}
function qi(e) {
    return [
        ...e
    ].map((t)=>{
        var n, r;
        return {
            name: `${t.columns}x${t.rows}`,
            columns: t.columns,
            rows: t.rows,
            maxTiles: t.columns * t.rows,
            minWidth: (n = t.minWidth) != null ? n : 0,
            minHeight: (r = t.minHeight) != null ? r : 0,
            orientation: t.orientation
        };
    }).sort((t, n)=>t.maxTiles !== n.maxTiles ? t.maxTiles - n.maxTiles : t.minWidth !== 0 || n.minWidth !== 0 ? t.minWidth - n.minWidth : t.minHeight !== 0 || n.minHeight !== 0 ? t.minHeight - n.minHeight : 0);
}
function Yo() {
    return typeof navigator < "u" && navigator.mediaDevices && !!navigator.mediaDevices.getDisplayMedia;
}
function qo(e, t) {
    var n;
    return se(Y({}, e), {
        receivedAtMediaTimestamp: (n = t.rtpTimestamp) != null ? n : 0,
        receivedAt: t.timestamp
    });
}
function Ko(e, t, n) {
    return [
        ...e,
        ...t
    ].reduceRight((r, i)=>(r.find((o)=>o.id === i.id) || r.unshift(i), r), []).slice(0 - n);
}
var An = [], On = {
    showChat: !1,
    unreadMessages: 0,
    showSettings: !1
};
function Ki(e) {
    return typeof e == "object";
}
function Go(e) {
    return Array.isArray(e) && e.filter(Ki).length > 0;
}
function kn(e, t) {
    return t.audioLevel - e.audioLevel;
}
function Ln(e, t) {
    return e.isSpeaking === t.isSpeaking ? 0 : e.isSpeaking ? -1 : 1;
}
function _n(e, t) {
    var n, r, i, o;
    return e.lastSpokeAt !== void 0 || t.lastSpokeAt !== void 0 ? ((r = (n = t.lastSpokeAt) == null ? void 0 : n.getTime()) != null ? r : 0) - ((o = (i = e.lastSpokeAt) == null ? void 0 : i.getTime()) != null ? o : 0) : 0;
}
function Be(e, t) {
    var n, r, i, o;
    return ((r = (n = e.joinedAt) == null ? void 0 : n.getTime()) != null ? r : 0) - ((o = (i = t.joinedAt) == null ? void 0 : i.getTime()) != null ? o : 0);
}
function Gi(e, t) {
    return j(e) ? j(t) ? 0 : -1 : j(t) ? 1 : 0;
}
function Qi(e, t) {
    const n = e.participant.isCameraEnabled, r = t.participant.isCameraEnabled;
    return n !== r ? n ? -1 : 1 : 0;
}
function Qo(e) {
    const t = [], n = [], r = [], i = [];
    e.forEach((a)=>{
        a.participant.isLocal && a.source === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Camera ? t.push(a) : a.source === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.ScreenShare ? n.push(a) : a.source === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Camera ? r.push(a) : i.push(a);
    });
    const o = Ji(n), s = Xi(r);
    return [
        ...t,
        ...o,
        ...s,
        ...i
    ];
}
function Ji(e) {
    const t = [], n = [];
    return e.forEach((i)=>{
        i.participant.isLocal ? t.push(i) : n.push(i);
    }), t.sort((i, o)=>Be(i.participant, o.participant)), n.sort((i, o)=>Be(i.participant, o.participant)), [
        ...n,
        ...t
    ];
}
function Xi(e) {
    const t = [], n = [];
    return e.forEach((r)=>{
        r.participant.isLocal ? t.push(r) : n.push(r);
    }), n.sort((r, i)=>r.participant.isSpeaking && i.participant.isSpeaking ? kn(r.participant, i.participant) : r.participant.isSpeaking !== i.participant.isSpeaking ? Ln(r.participant, i.participant) : r.participant.lastSpokeAt !== i.participant.lastSpokeAt ? _n(r.participant, i.participant) : j(r) !== j(i) ? Gi(r, i) : r.participant.isCameraEnabled !== i.participant.isCameraEnabled ? Qi(r, i) : Be(r.participant, i.participant)), [
        ...t,
        ...n
    ];
}
function Jo(e) {
    const t = [
        ...e
    ];
    t.sort((r, i)=>{
        if (r.isSpeaking && i.isSpeaking) return kn(r, i);
        if (r.isSpeaking !== i.isSpeaking) return Ln(r, i);
        if (r.lastSpokeAt !== i.lastSpokeAt) return _n(r, i);
        const o = r.videoTrackPublications.size > 0, s = i.videoTrackPublications.size > 0;
        return o !== s ? o ? -1 : 1 : Be(r, i);
    });
    const n = t.find((r)=>r.isLocal);
    if (n) {
        const r = t.indexOf(n);
        r >= 0 && (t.splice(r, 1), t.length > 0 ? t.splice(0, 0, n) : t.push(n));
    }
    return t;
}
function Zi(e, t) {
    return e.reduce((n, r, i)=>i % t === 0 ? [
            ...n,
            [
                r
            ]
        ] : [
            ...n.slice(0, -1),
            [
                ...n.slice(-1)[0],
                r
            ]
        ], []);
}
function Wt(e, t) {
    const n = Math.max(e.length, t.length);
    return new Array(n).fill([]).map((r, i)=>[
            e[i],
            t[i]
        ]);
}
function Ve(e, t, n) {
    return e.filter((r)=>!t.map((i)=>n(i)).includes(n(r)));
}
function ut(e) {
    return e.map((t)=>typeof t == "string" || typeof t == "number" ? `${t}` : N(t));
}
function eo(e, t) {
    return {
        dropped: Ve(e, t, N),
        added: Ve(t, e, N)
    };
}
function to(e) {
    return e.added.length !== 0 || e.dropped.length !== 0;
}
function lt(e, t) {
    const n = t.findIndex((r)=>N(r) === N(e));
    if (n === -1) throw new Error(`Element not part of the array: ${N(e)} not in ${ut(t)}`);
    return n;
}
function no(e, t, n) {
    const r = lt(e, n), i = lt(t, n);
    return n.splice(r, 1, t), n.splice(i, 1, e), n;
}
function ro(e, t) {
    const n = lt(e, t);
    return t.splice(n, 1), t;
}
function io(e, t) {
    return [
        ...t,
        e
    ];
}
function tt(e, t) {
    return Zi(e, t);
}
function Xo(e, t, n) {
    let r = oo(e, t);
    if (r.length < t.length) {
        const s = Ve(t, r, N);
        r = [
            ...r,
            ...s
        ];
    }
    const i = tt(r, n), o = tt(t, n);
    if (Wt(i, o).forEach(([s, a], c)=>{
        if (s && a) {
            const u = tt(r, n)[c], l = eo(u, a);
            to(l) && (_.debug(`Detected visual changes on page: ${c}, current: ${ut(s)}, next: ${ut(a)}`, {
                changes: l
            }), l.added.length === l.dropped.length && Wt(l.added, l.dropped).forEach(([f, v])=>{
                if (f && v) r = no(f, v, r);
                else throw new Error(`For a swap action we need a addition and a removal one is missing: ${f}, ${v}`);
            }), l.added.length === 0 && l.dropped.length > 0 && l.dropped.forEach((f)=>{
                r = ro(f, r);
            }), l.added.length > 0 && l.dropped.length === 0 && l.added.forEach((f)=>{
                r = io(f, r);
            }));
        }
    }), r.length > t.length) {
        const s = Ve(r, t, N);
        r = r.filter((a)=>!s.map(N).includes(N(a)));
    }
    return r;
}
function oo(e, t) {
    return e.map((n)=>{
        const r = t.find((i)=>// If the IDs match or ..
            N(n) === N(i) || // ... if the current item is a placeholder and the new item is the track reference can replace it.
            typeof n != "number" && Pe(n) && j(i) && ji(n, i));
        return r ?? n;
    });
}
function F(e) {
    return `${En}-${e}`;
}
function Zo(e) {
    const t = Bt(e), n = In(e.participant).pipe(O(()=>Bt(e)), D(t));
    return {
        className: F(e.source === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Camera || e.source === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.ScreenShare ? "participant-media-video" : "participant-media-audio"),
        trackObserver: n
    };
}
function Bt(e) {
    if (j(e)) return e.publication;
    {
        const { source: t, name: n, participant: r } = e;
        if (t && n) return r.getTrackPublications().find((i)=>i.source === t && i.trackName === n);
        if (n) return r.getTrackPublicationByName(n);
        if (t) return r.getTrackPublication(t);
        throw new Error("At least one of source and name needs to be defined");
    }
}
function fe(e, ...t) {
    return new k((r)=>{
        const i = ()=>{
            r.next(e);
        };
        return t.forEach((s)=>{
            e.on(s, i);
        }), ()=>{
            t.forEach((s)=>{
                e.off(s, i);
            });
        };
    }).pipe(D(e));
}
function we(e, t) {
    return new k((r)=>{
        const i = (...s)=>{
            r.next(s);
        };
        return e.on(t, i), ()=>{
            e.off(t, i);
        };
    });
}
function es(e) {
    return we(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ConnectionStateChanged).pipe(O(([t])=>t), D(e.state));
}
function ts(e) {
    return fe(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].RoomMetadataChanged, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ConnectionStateChanged).pipe(O((n)=>({
            name: n.name,
            metadata: n.metadata
        })));
}
function ns(e) {
    return we(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ActiveSpeakersChanged).pipe(O(([t])=>t));
}
function rs(e, t, n = !0) {
    const r = new k((o)=>{
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Room"].getLocalDevices(e, n).then((s)=>{
            o.next(s), o.complete();
        }).catch((s)=>{
            t == null || t(s), o.next([]), o.complete();
        });
    }), i = new k((o)=>{
        var s;
        const a = ()=>V(this, null, function*() {
                try {
                    const c = yield __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Room"].getLocalDevices(e, n);
                    o.next(c);
                } catch (c) {
                    t == null || t(c);
                }
            });
        if (typeof window < "u") {
            if (!window.isSecureContext) throw new Error("Accessing media devices is available only in secure contexts (HTTPS and localhost), in some or all supporting browsers. See: https://developer.mozilla.org/en-US/docs/Web/API/Navigator/mediaDevices");
            (s = navigator == null ? void 0 : navigator.mediaDevices) == null || s.addEventListener("devicechange", a);
        }
        return ()=>{
            var c;
            (c = navigator == null ? void 0 : navigator.mediaDevices) == null || c.removeEventListener("devicechange", a);
        };
    });
    return We(r, i);
}
function so(e) {
    return we(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].DataReceived);
}
function ao(e) {
    return fe(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].AudioPlaybackStatusChanged).pipe(O((n)=>({
            canPlayAudio: n.canPlaybackAudio
        })));
}
function co(e) {
    return fe(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].VideoPlaybackStatusChanged).pipe(O((n)=>({
            canPlayVideo: n.canPlaybackVideo
        })));
}
function uo(e, t) {
    return we(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ActiveDeviceChanged).pipe(Qe(([n])=>n === t), O(([n, r])=>(_.debug("activeDeviceObservable | RoomEvent.ActiveDeviceChanged", {
            kind: n,
            deviceId: r
        }), r)));
}
function is(e, t) {
    return we(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ParticipantEncryptionStatusChanged).pipe(Qe(([, n])=>(t == null ? void 0 : t.identity) === (n == null ? void 0 : n.identity) || !n && (t == null ? void 0 : t.identity) === e.localParticipant.identity), O(([n])=>n), D(t != null && t.isLocal ? t.isE2EEEnabled : !!(t != null && t.isEncrypted)));
}
function os(e) {
    return we(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].RecordingStatusChanged).pipe(O(([t])=>t), D(e.isRecording));
}
function xe(e, ...t) {
    return new k((r)=>{
        const i = ()=>{
            r.next(e);
        };
        return t.forEach((s)=>{
            e.on(s, i);
        }), ()=>{
            t.forEach((s)=>{
                e.off(s, i);
            });
        };
    }).pipe(D(e));
}
function In(e) {
    return xe(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackMuted, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackUnmuted, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].ParticipantPermissionsChanged, // ParticipantEvent.IsSpeakingChanged,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackPublished, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackUnpublished, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].LocalTrackPublished, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].LocalTrackUnpublished, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].MediaDevicesError, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackSubscriptionStatusChanged).pipe(O((n)=>{
        const { isMicrophoneEnabled: r, isCameraEnabled: i, isScreenShareEnabled: o } = n, s = n.getTrackPublication(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Microphone), a = n.getTrackPublication(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Camera);
        return {
            isCameraEnabled: i,
            isMicrophoneEnabled: r,
            isScreenShareEnabled: o,
            cameraTrack: a,
            microphoneTrack: s,
            participant: n
        };
    }));
}
function lo(e) {
    return e ? xe(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].ParticipantMetadataChanged, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].ParticipantNameChanged).pipe(O(({ name: n, identity: r, metadata: i })=>({
            name: n,
            identity: r,
            metadata: i
        })), D({
        name: e.name,
        identity: e.identity,
        metadata: e.metadata
    })) : void 0;
}
function fo(e) {
    return Je(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].ConnectionQualityChanged).pipe(O(([n])=>n), D(e.connectionQuality));
}
function Je(e, t) {
    return new k((r)=>{
        const i = (...s)=>{
            r.next(s);
        };
        return e.on(t, i), ()=>{
            e.off(t, i);
        };
    });
}
function po(e) {
    var t, n, r, i;
    return xe(e.participant, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackMuted, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackUnmuted, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackSubscribed, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].TrackUnsubscribed, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].LocalTrackPublished, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].LocalTrackUnpublished).pipe(O((o)=>{
        var s, a;
        const c = (s = e.publication) != null ? s : o.getTrackPublication(e.source);
        return (a = c == null ? void 0 : c.isMuted) != null ? a : !0;
    }), D((i = (r = (t = e.publication) == null ? void 0 : t.isMuted) != null ? r : (n = e.participant.getTrackPublication(e.source)) == null ? void 0 : n.isMuted) != null ? i : !0));
}
function ss(e) {
    return Je(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].IsSpeakingChanged).pipe(O(([t])=>t));
}
function as(e, t = {}) {
    var n;
    let r;
    const i = new k((c)=>(r = c, ()=>a.unsubscribe())).pipe(D(Array.from(e.remoteParticipants.values()))), o = (n = t.additionalRoomEvents) != null ? n : Cn, s = Array.from(/* @__PURE__ */ new Set([
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ParticipantConnected,
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ParticipantDisconnected,
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ConnectionStateChanged,
        ...o
    ])), a = fe(e, ...s).subscribe((c)=>r == null ? void 0 : r.next(Array.from(c.remoteParticipants.values())));
    return e.remoteParticipants.size > 0 && (r == null || r.next(Array.from(e.remoteParticipants.values()))), i;
}
function cs(e, t, n = {}) {
    var r;
    const i = (r = n.additionalEvents) != null ? r : Pn;
    return fe(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ParticipantConnected, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ParticipantDisconnected, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ConnectionStateChanged).pipe(Tn((s)=>{
        const a = s.getParticipantByIdentity(t);
        return a ? xe(a, ...i) : new k((c)=>c.next(void 0));
    }), D(e.getParticipantByIdentity(t)));
}
function us(e) {
    return Je(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].ParticipantPermissionsChanged).pipe(O(()=>e.permissions), D(e.permissions));
}
function ls(e, { kind: t, identity: n }, r = {}) {
    var i;
    const o = (i = r.additionalEvents) != null ? i : Pn, s = (c)=>{
        let u = !0;
        return t && (u = u && c.kind === t), n && (u = u && c.identity === n), u;
    };
    return fe(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ParticipantConnected, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ParticipantDisconnected, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ConnectionStateChanged).pipe(Tn((c)=>{
        const u = Array.from(c.remoteParticipants.values()).find((l)=>s(l));
        return u ? xe(u, ...o) : new k((l)=>l.next(void 0));
    }), D(Array.from(e.remoteParticipants.values()).find((c)=>s(c))));
}
function fs(e) {
    return typeof e > "u" ? new k() : Je(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantEvent"].AttributesChanged).pipe(O(([t])=>({
            changed: t,
            attributes: e.attributes
        })), D({
        changed: e.attributes,
        attributes: e.attributes
    }));
}
function ds(e, t, n, r, i) {
    const { localParticipant: o } = t, s = (f, v)=>{
        let d = !1;
        switch(f){
            case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Camera:
                d = v.isCameraEnabled;
                break;
            case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Microphone:
                d = v.isMicrophoneEnabled;
                break;
            case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.ScreenShare:
                d = v.isScreenShareEnabled;
                break;
        }
        return d;
    }, a = In(o).pipe(O((f)=>s(e, f.participant)), D(s(e, o))), c = new ee(), u = (f, v)=>V(this, null, function*() {
            try {
                switch(v ?? (v = n), c.next(!0), e){
                    case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Camera:
                        return yield o.setCameraEnabled(f ?? !o.isCameraEnabled, v, r), o.isCameraEnabled;
                    case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Microphone:
                        return yield o.setMicrophoneEnabled(f ?? !o.isMicrophoneEnabled, v, r), o.isMicrophoneEnabled;
                    case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.ScreenShare:
                        return yield o.setScreenShareEnabled(f ?? !o.isScreenShareEnabled, v, r), o.isScreenShareEnabled;
                    default:
                        throw new TypeError("Tried to toggle unsupported source");
                }
            } catch (d) {
                if (i && d instanceof Error) {
                    i == null || i(d);
                    return;
                } else throw d;
            } finally{
                c.next(!1);
            }
        });
    return {
        className: F("button"),
        toggle: u,
        enabledObserver: a,
        pendingObserver: c.asObservable()
    };
}
function ps() {
    let e = !1;
    const t = new ee(), n = new ee(), r = (o)=>V(this, null, function*() {
            n.next(!0), e = o ?? !e, t.next(e), n.next(!1);
        });
    return {
        className: F("button"),
        toggle: r,
        enabledObserver: t.asObservable(),
        pendingObserver: n.asObservable()
    };
}
function hs(e, t, n) {
    const r = new ln(void 0), i = uo(t, e), o = (a, ...c)=>V(this, [
            a,
            ...c
        ], function*(u, l = {}) {
            var f, v, d;
            if (t) {
                _.debug(`Switching active device of kind "${e}" with id ${u}.`), yield t.switchActiveDevice(e, u, l.exact);
                const m = (f = t.getActiveDevice(e)) != null ? f : u;
                m !== u && u !== "default" && _.info(`We tried to select the device with id (${u}), but the browser decided to select the device with id (${m}) instead.`);
                let p;
                e === "audioinput" ? p = (v = t.localParticipant.getTrackPublication(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Microphone)) == null ? void 0 : v.track : e === "videoinput" && (p = (d = t.localParticipant.getTrackPublication(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Camera)) == null ? void 0 : d.track);
                const g = u === "default" && !p || u === "default" && (p == null ? void 0 : p.mediaStreamTrack.label.startsWith("Default"));
                r.next(g ? u : m);
            }
        });
    return {
        className: F("media-device-select"),
        activeDeviceObservable: i,
        setActiveMediaDevice: o
    };
}
function vs(e) {
    const t = (r)=>{
        e.disconnect(r);
    };
    return {
        className: F("disconnect-button"),
        disconnect: t
    };
}
function ms(e) {
    const t = F("connection-quality"), n = fo(e);
    return {
        className: t,
        connectionQualityObserver: n
    };
}
function gs(e) {
    let t = "track-muted-indicator-camera";
    switch(e.source){
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Camera:
            t = "track-muted-indicator-camera";
            break;
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Microphone:
            t = "track-muted-indicator-microphone";
            break;
    }
    const n = F(t), r = po(e);
    return {
        className: n,
        mediaMutedObserver: r
    };
}
function bs(e) {
    return {
        className: "lk-participant-name",
        infoObserver: lo(e)
    };
}
function ys() {
    return {
        className: F("participant-tile")
    };
}
var ho = {
    CHAT: "lk.chat",
    TRANSCRIPTION: "lk.transcription"
}, vo = {
    CHAT: "lk-chat-topic"
};
function Mn(e, t) {
    return V(this, arguments, function*(n, r, i = {}) {
        const { reliable: o, destinationIdentities: s, topic: a } = i;
        yield n.publishData(r, {
            destinationIdentities: s,
            topic: a,
            reliable: o
        });
    });
}
function mo(e, t, n) {
    const r = Array.isArray(t) ? t : [
        t
    ], i = so(e).pipe(Qe(([, , , c])=>t === void 0 || c !== void 0 && r.includes(c)), O(([c, u, , l])=>{
        const f = {
            payload: c,
            topic: l,
            from: u
        };
        return n == null || n(f), f;
    }));
    let o;
    const s = new k((c)=>{
        o = c;
    });
    return {
        messageObservable: i,
        isSendingObservable: s,
        send: (c, ...u)=>V(this, [
                c,
                ...u
            ], function*(l, f = {}) {
                o.next(!0);
                try {
                    yield Mn(e.localParticipant, l, Y({
                        topic: r[0]
                    }, f));
                } finally{
                    o.next(!1);
                }
            })
    };
}
var Ie = /* @__PURE__ */ new WeakMap();
function go(e) {
    return e.ignoreLegacy == !0;
}
var bo = (e)=>JSON.parse(new TextDecoder().decode(e)), yo = (e)=>new TextEncoder().encode(JSON.stringify(e));
function ws(e, t) {
    var n, r, i, o, s, a;
    const c = ()=>{
        var b, S, C;
        return ((b = e.serverInfo) == null ? void 0 : b.edition) === 1 || !!((S = e.serverInfo) != null && S.version) && (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["compareVersions"])((C = e.serverInfo) == null ? void 0 : C.version, "1.8.2") > 0;
    }, u = new ee(), l = (n = t == null ? void 0 : t.channelTopic) != null ? n : ho.CHAT, f = (r = t == null ? void 0 : t.channelTopic) != null ? r : vo.CHAT;
    let v = !1;
    Ie.has(e) || (v = !0);
    const d = (i = Ie.get(e)) != null ? i : /* @__PURE__ */ new Map(), m = (o = d.get(l)) != null ? o : new ee();
    d.set(l, m), Ie.set(e, d);
    const p = (s = t == null ? void 0 : t.messageDecoder) != null ? s : bo;
    if (v) {
        e.registerTextStreamHandler(l, (S, C)=>V(this, null, function*() {
                const { id: $, timestamp: I } = S.info;
                Ge(S).pipe(ct((T, L)=>T + L), O((T)=>({
                        id: $,
                        timestamp: I,
                        message: T,
                        from: e.getParticipantByIdentity(C.identity)
                    }))).subscribe({
                    next: (T)=>m.next(T)
                });
            }));
        const { messageObservable: b } = mo(e, [
            f
        ]);
        b.pipe(O((S)=>{
            const C = p(S.payload);
            return go(C) ? void 0 : se(Y({}, C), {
                from: S.from
            });
        }), Qe((S)=>!!S), Nt(u)).subscribe(m);
    }
    const g = m.pipe(ct((b, S)=>{
        if ("id" in S && b.find((C)=>{
            var $, I;
            return (($ = C.from) == null ? void 0 : $.identity) === ((I = S.from) == null ? void 0 : I.identity) && C.id === S.id;
        })) {
            const C = b.findIndex(($)=>$.id === S.id);
            if (C > -1) {
                const $ = b[C];
                b[C] = se(Y({}, S), {
                    timestamp: $.timestamp,
                    editTimestamp: S.timestamp
                });
            }
            return [
                ...b
            ];
        }
        return [
            ...b,
            S
        ];
    }, []), Nt(u)), h = new ln(!1), x = (a = t == null ? void 0 : t.messageEncoder) != null ? a : yo, E = (b, S)=>V(this, null, function*() {
            var C;
            S || (S = {}), (C = S.topic) != null || (S.topic = l), h.next(!0);
            try {
                const I = {
                    id: (yield e.localParticipant.sendText(b, S)).id,
                    timestamp: Date.now(),
                    message: b
                }, z = se(Y({}, I), {
                    attachedFiles: S.attachments
                }), T = se(Y({}, z), {
                    from: e.localParticipant,
                    attributes: S.attributes
                });
                m.next(T);
                const L = x(se(Y({}, I), {
                    ignoreLegacy: c()
                }));
                try {
                    yield Mn(e.localParticipant, L, {
                        reliable: !0,
                        topic: f
                    });
                } catch (Z) {
                    _.info("could not send message in legacy chat format", Z);
                }
                return T;
            } finally{
                h.next(!1);
            }
        });
    function P() {
        u.next(), u.complete(), m.complete(), Ie.delete(e), e.unregisterTextStreamHandler(l);
    }
    return e.once(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].Disconnected, P), {
        messageObservable: g,
        isSendingObservable: h,
        send: E
    };
}
function xs() {
    const e = (n)=>V(this, null, function*() {
            _.info("Start Audio for room: ", n), yield n.startAudio();
        });
    return {
        className: F("start-audio-button"),
        roomAudioPlaybackAllowedObservable: ao,
        handleStartAudioPlayback: e
    };
}
function Ss() {
    const e = (n)=>V(this, null, function*() {
            _.info("Start Video for room: ", n), yield n.startVideo();
        });
    return {
        className: F("start-audio-button"),
        roomVideoPlaybackAllowedObservable: co,
        handleStartVideoPlayback: e
    };
}
function Ts() {
    return {
        className: [
            F("button"),
            F("chat-toggle")
        ].join(" ")
    };
}
function Es() {
    return {
        className: [
            F("button"),
            F("focus-toggle-button")
        ].join(" ")
    };
}
function Cs() {
    return {
        className: "lk-clear-pin-button lk-button"
    };
}
function Ps() {
    return {
        className: "lk-room-container"
    };
}
function Vt(e, t, n = !0) {
    const i = [
        e.localParticipant,
        ...Array.from(e.remoteParticipants.values())
    ], o = [];
    return i.forEach((s)=>{
        t.forEach((a)=>{
            const c = Array.from(s.trackPublications.values()).filter((u)=>u.source === a && // either return all or only the ones that are subscribed
                (!n || u.track)).map((u)=>({
                    participant: s,
                    publication: u,
                    source: u.source
                }));
            o.push(...c);
        });
    }), {
        trackReferences: o,
        participants: i
    };
}
function Ht(e, t, n = !1) {
    const { sources: r, kind: i, name: o } = t;
    return Array.from(e.trackPublications.values()).filter((a)=>(!r || r.includes(a.source)) && (!i || a.kind === i) && (!o || a.trackName === o) && // either return all or only the ones that are subscribed
        (!n || a.track)).map((a)=>({
            participant: e,
            publication: a,
            source: a.source
        }));
}
function As(e, t, n) {
    var r, i;
    const o = (r = n.additionalRoomEvents) != null ? r : Cn, s = (i = n.onlySubscribed) != null ? i : !0, a = Array.from(/* @__PURE__ */ new Set([
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ParticipantConnected,
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ParticipantDisconnected,
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].ConnectionStateChanged,
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].LocalTrackPublished,
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].LocalTrackUnpublished,
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].TrackPublished,
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].TrackUnpublished,
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].TrackSubscriptionStatusChanged,
        ...o
    ]).values());
    return fe(e, ...a).pipe(O((u)=>{
        const l = Vt(u, t, s);
        return _.debug(`TrackReference[] was updated. (length ${l.trackReferences.length})`, l), l;
    }), D(Vt(e, t, s)));
}
function Os(e, t) {
    return xe(e, ...Hi).pipe(O((r)=>{
        const i = Ht(r, t);
        return _.debug(`TrackReference[] was updated. (length ${i.length})`, i), i;
    }), D(Ht(e, t)));
}
function Rn(e, t) {
    return new k((r)=>{
        const i = (...s)=>{
            r.next(s);
        };
        return e.on(t, i), ()=>{
            e.off(t, i);
        };
    });
}
function ks(e) {
    return Rn(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["TrackEvent"].TranscriptionReceived);
}
function Ls(e) {
    return Rn(e, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["TrackEvent"].TimeSyncUpdate).pipe(O(([t])=>t));
}
function _s(e, t = 1e3) {
    if (e === null) return Dt(!1);
    const n = at(e, "mousemove", {
        passive: !0
    }).pipe(O(()=>!0)), r = n.pipe(hi({
        each: t,
        with: ()=>We(Dt(!1), r.pipe(Ii(n)))
    }), ki());
    return r;
}
function wo(e, t) {
    if (typeof localStorage > "u") {
        _.error("Local storage is not available.");
        return;
    }
    try {
        if (t) {
            const n = Object.fromEntries(Object.entries(t).filter(([, r])=>r !== ""));
            localStorage.setItem(e, JSON.stringify(n));
        }
    } catch (n) {
        _.error(`Error setting item to local storage: ${n}`);
    }
}
function xo(e) {
    if (typeof localStorage > "u") {
        _.error("Local storage is not available.");
        return;
    }
    try {
        const t = localStorage.getItem(e);
        if (!t) {
            _.warn(`Item with key ${e} does not exist in local storage.`);
            return;
        }
        return JSON.parse(t);
    } catch (t) {
        _.error(`Error getting item from local storage: ${t}`);
        return;
    }
}
function So(e) {
    return {
        load: ()=>xo(e),
        save: (t)=>wo(e, t)
    };
}
var To = `${En}-user-choices`, Te = {
    videoEnabled: !0,
    audioEnabled: !0,
    videoDeviceId: "default",
    audioDeviceId: "default",
    username: ""
}, { load: Eo, save: Co } = So(To);
function Is(e, t = !1) {
    t !== !0 && Co(e);
}
function Ms(e, t = !1) {
    var n, r, i, o, s;
    const a = {
        videoEnabled: (n = e == null ? void 0 : e.videoEnabled) != null ? n : Te.videoEnabled,
        audioEnabled: (r = e == null ? void 0 : e.audioEnabled) != null ? r : Te.audioEnabled,
        videoDeviceId: (i = e == null ? void 0 : e.videoDeviceId) != null ? i : Te.videoDeviceId,
        audioDeviceId: (o = e == null ? void 0 : e.audioDeviceId) != null ? o : Te.audioDeviceId,
        username: (s = e == null ? void 0 : e.username) != null ? s : Te.username
    };
    if (t) return a;
    {
        const c = Eo();
        return Y(Y({}, a), c ?? {});
    }
}
var nt = null, rt = null, Po = 0;
function zt() {
    return nt || (nt = /* @__PURE__ */ new Map()), nt;
}
function Ao() {
    return rt || (rt = /* @__PURE__ */ new WeakMap()), rt;
}
function Oo(e, t) {
    const n = Ao();
    let r = n.get(e);
    return r || (r = `room_${Po++}`, n.set(e, r)), `${r}:${t}`;
}
function Rs(e, t) {
    const n = Oo(e, t), r = zt(), i = r.get(n);
    if (i) return i;
    const o = new ee(), s = [], a = "lk.segment_id";
    e.registerTextStreamHandler(t, (u, l)=>V(this, null, function*() {
            var f;
            const v = Ge(u).pipe(ct((m, p)=>m + p, "")), d = !!((f = u.info.attributes) != null && f[a]);
            v.subscribe((m)=>{
                const p = s.findIndex((g)=>{
                    var h, x;
                    return g.streamInfo.id === u.info.id || d && ((h = g.streamInfo.attributes) == null ? void 0 : h[a]) === ((x = u.info.attributes) == null ? void 0 : x[a]);
                });
                p !== -1 ? (s[p] = se(Y({}, s[p]), {
                    text: m
                }), o.next([
                    ...s
                ])) : (s.push({
                    text: m,
                    participantInfo: l,
                    streamInfo: u.info
                }), o.next([
                    ...s
                ]));
            });
        }));
    const c = o.asObservable().pipe(_i());
    return r.set(n, c), e.once(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].Disconnected, ()=>{
        e.unregisterTextStreamHandler(t), o.complete(), zt().delete(n);
    }), c;
}
function Dn(e, t) {
    if (t.msg === "show_chat") return {
        ...e,
        showChat: !0,
        unreadMessages: 0
    };
    if (t.msg === "hide_chat") return {
        ...e,
        showChat: !1
    };
    if (t.msg === "toggle_chat") {
        const n = {
            ...e,
            showChat: !e.showChat
        };
        return n.showChat === !0 && (n.unreadMessages = 0), n;
    } else return t.msg === "unread_msg" ? {
        ...e,
        unreadMessages: t.count
    } : t.msg === "toggle_settings" ? {
        ...e,
        showSettings: !e.showSettings
    } : {
        ...e
    };
}
function $n(e, t) {
    return t.msg === "set_pin" ? [
        t.trackReference
    ] : t.msg === "clear_pin" ? [] : {
        ...e
    };
}
const Nn = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createContext"])(void 0);
function Ds() {
    const e = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useContext"])(Nn);
    if (!e) throw Error("Tried to access LayoutContext context outside a LayoutContextProvider provider.");
    return e;
}
function $s(e) {
    const t = ko();
    if (e ?? (e = t), !e) throw Error("Tried to access LayoutContext context outside a LayoutContextProvider provider.");
    return e;
}
function Ns() {
    const [e, t] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useReducer"])($n, An), [n, r] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useReducer"])(Dn, On);
    return {
        pin: {
            dispatch: t,
            state: e
        },
        widget: {
            dispatch: r,
            state: n
        }
    };
}
function Fs(e) {
    const [t, n] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useReducer"])($n, An), [r, i] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useReducer"])(Dn, On);
    return e ?? {
        pin: {
            dispatch: n,
            state: t
        },
        widget: {
            dispatch: i,
            state: r
        }
    };
}
function ko() {
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useContext"])(Nn);
}
const Fn = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createContext"])(void 0);
function Us() {
    const e = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useContext"])(Fn);
    if (!e) throw Error("tried to access track context outside of track context provider");
    return e;
}
function Un() {
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useContext"])(Fn);
}
function js(e) {
    const t = Un(), n = e ?? t;
    if (!n) throw new Error("No TrackRef, make sure you are inside a TrackRefContext or pass the TrackRef explicitly");
    return n;
}
const jn = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createContext"])(void 0);
function Ws() {
    const e = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useContext"])(jn);
    if (!e) throw Error("tried to access participant context outside of participant context provider");
    return e;
}
function Lo() {
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useContext"])(jn);
}
function Bs(e) {
    const t = Lo(), n = Un(), r = e ?? t ?? (n == null ? void 0 : n.participant);
    if (!r) throw new Error("No participant provided, make sure you are inside a participant context or pass the participant explicitly");
    return r;
}
const Wn = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createContext"])(void 0);
function Vs() {
    const e = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useContext"])(Wn);
    if (!e) throw Error("tried to access room context outside of livekit room component");
    return e;
}
function _o() {
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useContext"])(Wn);
}
function Hs(e) {
    const t = _o(), n = e ?? t;
    if (!n) throw new Error("No room provided, make sure you are inside a Room context or pass the room explicitly");
    return n;
}
const Io = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createContext"])(void 0);
function zs(e) {
    const t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useContext"])(Io);
    if (e === !0) {
        if (t) return t;
        throw Error("tried to access feature context, but none is present");
    }
    return t;
}
;
 //# sourceMappingURL=contexts-CH4CA2DN.mjs.map
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/room-DhBnHppi.mjs [app-client] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "L": (()=>W),
    "a": (()=>G),
    "c": (()=>H),
    "m": (()=>M),
    "r": (()=>T),
    "u": (()=>$),
    "w": (()=>Q)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$build$2f$polyfills$2f$process$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/build/polyfills/process.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/contexts-CH4CA2DN.mjs [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-client@2.15.2_@types+dom-mediacapture-record@1.0.22/node_modules/livekit-client/dist/livekit-client.esm.mjs [app-client] (ecmascript)");
;
;
;
function L(n) {
    var e, o, t = "";
    if (typeof n == "string" || typeof n == "number") t += n;
    else if (typeof n == "object") if (Array.isArray(n)) {
        var r = n.length;
        for(e = 0; e < r; e++)n[e] && (o = L(n[e])) && (t && (t += " "), t += o);
    } else for(o in n)n[o] && (t && (t += " "), t += o);
    return t;
}
function A() {
    for(var n, e, o = 0, t = "", r = arguments.length; o < r; o++)(n = arguments[o]) && (e = L(n)) && (t && (t += " "), t += e);
    return t;
}
function I(...n) {
    return (...e)=>{
        for (const o of n)if (typeof o == "function") try {
            o(...e);
        } catch (t) {
            console.error(t);
        }
    };
}
function M(...n) {
    const e = {
        ...n[0]
    };
    for(let o = 1; o < n.length; o++){
        const t = n[o];
        for(const r in t){
            const d = e[r], a = t[r];
            typeof d == "function" && typeof a == "function" && // This is a lot faster than a regex.
            r[0] === "o" && r[1] === "n" && r.charCodeAt(2) >= /* 'A' */ 65 && r.charCodeAt(2) <= /* 'Z' */ 90 ? e[r] = I(d, a) : (r === "className" || r === "UNSAFE_className") && typeof d == "string" && typeof a == "string" ? e[r] = A(d, a) : e[r] = a !== void 0 ? a : d;
        }
    }
    return e;
}
function J(n) {
    return n !== void 0;
}
function G(...n) {
    return M(...n.filter(J));
}
function H(n, e, o) {
    return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Children"].map(n, (t)=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["isValidElement"])(t) && __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Children"].only(n) ? (t.props.className && (e ?? (e = {}), e.className = A(t.props.className, e.className), e.style = {
            ...t.props.style,
            ...e.style
        }), (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["cloneElement"])(t, {
            ...e,
            key: o
        })) : t);
}
function Q(n) {
    var e, o;
    if (typeof window < "u" && typeof __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$build$2f$polyfills$2f$process$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["default"] < "u" && // eslint-disable-next-line turbo/no-undeclared-env-vars
    (((e = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$build$2f$polyfills$2f$process$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["default"] == null ? void 0 : __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$build$2f$polyfills$2f$process$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["default"].env) == null ? void 0 : e.NODE_ENV) === "dev" || // eslint-disable-next-line turbo/no-undeclared-env-vars
    ((o = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$build$2f$polyfills$2f$process$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["default"] == null ? void 0 : __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$build$2f$polyfills$2f$process$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["default"].env) == null ? void 0 : o.NODE_ENV) === "development")) {
        const t = document.querySelector(".lk-room-container");
        t && !getComputedStyle(t).getPropertyValue("--lk-has-imported-styles") && __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].warn("It looks like you're not using the `@livekit/components-styles package`. To render the UI with the default styling, please import it in your layout or page.");
    }
}
function T(n, e) {
    return n === "processor" && e && typeof e == "object" && "name" in e ? e.name : n === "e2ee" && e ? "e2ee-enabled" : e;
}
const q = {
    connect: !0,
    audio: !1,
    video: !1
};
function $(n) {
    const { token: e, serverUrl: o, options: t, room: r, connectOptions: d, connect: a, audio: p, video: y, screen: g, onConnected: v, onDisconnected: h, onError: c, onMediaDeviceFailure: b, onEncryptionError: E, simulateParticipants: w, ...N } = {
        ...q,
        ...n
    };
    t && r && __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].warn("when using a manually created room, the options object will be ignored. set the desired options directly when creating the room instead.");
    const [s, O] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(), C = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(a);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "$.useEffect": ()=>{
            O(r ?? new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Room"](t));
        }
    }["$.useEffect"], [
        r,
        JSON.stringify(t, T)
    ]);
    const F = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "$.useMemo[F]": ()=>{
            const { className: m } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["s"])();
            return M(N, {
                className: m
            });
        }
    }["$.useMemo[F]"], [
        N
    ]);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "$.useEffect": ()=>{
            if (!s) return;
            const m = {
                "$.useEffect.m": ()=>{
                    const f = s.localParticipant;
                    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].debug("trying to publish local tracks"), Promise.all([
                        f.setMicrophoneEnabled(!!p, typeof p != "boolean" ? p : void 0),
                        f.setCameraEnabled(!!y, typeof y != "boolean" ? y : void 0),
                        f.setScreenShareEnabled(!!g, typeof g != "boolean" ? g : void 0)
                    ]).catch({
                        "$.useEffect.m": (R)=>{
                            __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].warn(R), c == null || c(R);
                        }
                    }["$.useEffect.m"]);
                }
            }["$.useEffect.m"], P = {
                "$.useEffect.P": (f, R)=>{
                    const K = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["MediaDeviceFailure"].getFailure(f);
                    b == null || b(K, R);
                }
            }["$.useEffect.P"], S = {
                "$.useEffect.S": (f)=>{
                    E == null || E(f);
                }
            }["$.useEffect.S"], k = {
                "$.useEffect.k": (f)=>{
                    h == null || h(f);
                }
            }["$.useEffect.k"], D = {
                "$.useEffect.D": ()=>{
                    v == null || v();
                }
            }["$.useEffect.D"];
            return s.on(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].SignalConnected, m).on(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].MediaDevicesError, P).on(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].EncryptionError, S).on(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].Disconnected, k).on(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].Connected, D), ({
                "$.useEffect": ()=>{
                    s.off(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].SignalConnected, m).off(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].MediaDevicesError, P).off(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].EncryptionError, S).off(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].Disconnected, k).off(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].Connected, D);
                }
            })["$.useEffect"];
        }
    }["$.useEffect"], [
        s,
        p,
        y,
        g,
        c,
        E,
        b,
        v,
        h
    ]), (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "$.useEffect": ()=>{
            if (s) {
                if (w) {
                    s.simulateParticipants({
                        participants: {
                            count: w
                        },
                        publish: {
                            audio: !0,
                            useRealTracks: !0
                        }
                    });
                    return;
                }
                if (a) {
                    if (C.current = !0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].debug("connecting"), !e) {
                        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].debug("no token yet");
                        return;
                    }
                    if (!o) {
                        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].warn("no livekit url provided"), c == null || c(Error("no livekit url provided"));
                        return;
                    }
                    s.connect(o, e, d).catch({
                        "$.useEffect": (m)=>{
                            __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].warn(m), C.current === !0 && (c == null || c(m));
                        }
                    }["$.useEffect"]);
                } else __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].debug("disconnecting because connect is false"), C.current = !1, s.disconnect();
            }
        }
    }["$.useEffect"], [
        a,
        e,
        JSON.stringify(d),
        s,
        c,
        o,
        w
    ]), (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "$.useEffect": ()=>{
            if (s) return ({
                "$.useEffect": ()=>{
                    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].info("disconnecting on onmount"), s.disconnect();
                }
            })["$.useEffect"];
        }
    }["$.useEffect"], [
        s
    ]), {
        room: s,
        htmlProps: F
    };
}
const W = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function(e, o) {
    const { room: t, htmlProps: r } = $(e);
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("div", {
        ref: o,
        ...r
    }, t && /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["R"].Provider, {
        value: t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["L"].Provider, {
        value: e.featureFlags
    }, e.children)));
});
;
 //# sourceMappingURL=room-DhBnHppi.mjs.map
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "A": (()=>wt),
    "B": (()=>Nt),
    "C": (()=>Ke),
    "D": (()=>ht),
    "E": (()=>Pt),
    "F": (()=>Ye),
    "G": (()=>q),
    "H": (()=>Ze),
    "I": (()=>It),
    "J": (()=>Dt),
    "K": (()=>Rt),
    "L": (()=>Ft),
    "M": (()=>_),
    "N": (()=>_t),
    "O": (()=>tt),
    "P": (()=>Jt),
    "Q": (()=>Qt),
    "R": (()=>Xt),
    "S": (()=>Zt),
    "T": (()=>J),
    "U": (()=>ft),
    "V": (()=>en),
    "W": (()=>pt),
    "X": (()=>Kt),
    "Y": (()=>tn),
    "Z": (()=>bt),
    "_": (()=>nn),
    "a": (()=>O),
    "b": (()=>Ut),
    "c": (()=>kt),
    "d": (()=>Tt),
    "e": (()=>At),
    "f": (()=>zt),
    "g": (()=>Bt),
    "h": (()=>Wt),
    "i": (()=>yt),
    "j": (()=>b),
    "k": (()=>xt),
    "l": (()=>Lt),
    "m": (()=>jt),
    "n": (()=>Et),
    "o": (()=>Ot),
    "p": (()=>qt),
    "q": (()=>Xe),
    "r": (()=>et),
    "s": (()=>Yt),
    "t": (()=>Gt),
    "u": (()=>Mt),
    "v": (()=>Vt),
    "w": (()=>$t),
    "x": (()=>Ht),
    "y": (()=>Ct),
    "z": (()=>vt)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/contexts-CH4CA2DN.mjs [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/room-DhBnHppi.mjs [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-client@2.15.2_@types+dom-mediacapture-record@1.0.22/node_modules/livekit-client/dist/livekit-client.esm.mjs [app-client] (ecmascript)");
;
;
;
;
const Qe = (e)=>{
    const t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(e);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Qe.useEffect": ()=>{
            t.current = e;
        }
    }["Qe.useEffect"]), t;
};
function $e(e, t) {
    const n = je(), s = Qe(t);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useLayoutEffect"])({
        "$e.useLayoutEffect": ()=>{
            let a = !1;
            const c = e.current;
            if (!c) return;
            function o(u, i) {
                a || s.current(u, i);
            }
            return n == null || n.subscribe(c, o), ({
                "$e.useLayoutEffect": ()=>{
                    a = !0, n == null || n.unsubscribe(c, o);
                }
            })["$e.useLayoutEffect"];
        }
    }["$e.useLayoutEffect"], [
        e.current,
        n,
        s
    ]), n == null ? void 0 : n.observer;
}
function He() {
    let e = !1, t = [];
    const n = /* @__PURE__ */ new Map();
    if (typeof window > "u") return;
    const s = new ResizeObserver((a, c)=>{
        t = t.concat(a), e || window.requestAnimationFrame(()=>{
            const o = /* @__PURE__ */ new Set();
            for(let u = 0; u < t.length; u++){
                if (o.has(t[u].target)) continue;
                o.add(t[u].target);
                const i = n.get(t[u].target);
                i == null || i.forEach((d)=>d(t[u], c));
            }
            t = [], e = !1;
        }), e = !0;
    });
    return {
        observer: s,
        subscribe (a, c) {
            s.observe(a);
            const o = n.get(a) ?? [];
            o.push(c), n.set(a, o);
        },
        unsubscribe (a, c) {
            const o = n.get(a) ?? [];
            if (o.length === 1) {
                s.unobserve(a), n.delete(a);
                return;
            }
            const u = o.indexOf(c);
            u !== -1 && o.splice(u, 1), n.set(a, o);
        }
    };
}
let D;
const je = ()=>D || (D = He()), Xe = (e)=>{
    const [t, n] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])({
        width: 0,
        height: 0
    });
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useLayoutEffect"])({
        "Xe.useLayoutEffect": ()=>{
            if (e.current) {
                const { width: a, height: c } = e.current.getBoundingClientRect();
                n({
                    width: a,
                    height: c
                });
            }
        }
    }["Xe.useLayoutEffect"], [
        e.current
    ]);
    const s = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "Xe.useCallback[s]": (a)=>n(a.contentRect)
    }["Xe.useCallback[s]"], []);
    return $e(e, s), t;
};
function b(e, t, n = !0) {
    const [s, a] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(t);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "b.useEffect": ()=>{
            if (n && a(t), typeof window > "u" || !e) return;
            const c = e.subscribe(a);
            return ({
                "b.useEffect": ()=>c.unsubscribe()
            })["b.useEffect"];
        }
    }["b.useEffect"], [
        e,
        n
    ]), s;
}
function vt(e) {
    const t = (c)=>typeof window < "u" ? window.matchMedia(c).matches : !1, [n, s] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(t(e));
    function a() {
        s(t(e));
    }
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "vt.useEffect": ()=>{
            const c = window.matchMedia(e);
            return a(), c.addListener ? c.addListener(a) : c.addEventListener("change", a), ({
                "vt.useEffect": ()=>{
                    c.removeListener ? c.removeListener(a) : c.removeEventListener("change", a);
                }
            })["vt.useEffect"];
        }
    }["vt.useEffect"], [
        e
    ]), n;
}
function ht(e) {
    const t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["u"])(e), n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "ht.useCallback[n]": async ()=>{
            await t.startAudio();
        }
    }["ht.useCallback[n]"], [
        t
    ]), s = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "ht.useMemo[s]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["r"])(t)
    }["ht.useMemo[s]"], [
        t
    ]), { canPlayAudio: a } = b(s, {
        canPlayAudio: t.canPlaybackAudio
    });
    return {
        canPlayAudio: a,
        startAudio: n
    };
}
function Mt(e) {
    const { state: t, dispatch: n } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a"])().pin;
    return {
        buttonProps: (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
            "Mt.useMemo": ()=>{
                const { className: a } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["b"])();
                return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["m"])(e, {
                    className: a,
                    disabled: !(t != null && t.length),
                    onClick: {
                        "Mt.useMemo": ()=>{
                            n && n({
                                msg: "clear_pin"
                            });
                        }
                    }["Mt.useMemo"]
                });
            }
        }["Mt.useMemo"], [
            e,
            n,
            t
        ])
    };
}
function yt(e = {}) {
    const t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["c"])(e.participant), { className: n, connectionQualityObserver: s } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "yt.useMemo": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["d"])(t)
    }["yt.useMemo"], [
        t
    ]), a = b(s, t.connectionQuality);
    return {
        className: n,
        quality: a
    };
}
function O(e) {
    const t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["u"])(e), n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "O.useMemo[n]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["e"])(t)
    }["O.useMemo[n]"], [
        t
    ]);
    return b(n, t.state);
}
function Pt(e, t) {
    const n = typeof e == "function" ? e : t, s = typeof e == "string" ? e : void 0, a = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["f"])(), { send: c, messageObservable: o, isSendingObservable: u } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Pt.useMemo": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["g"])(a, s, n)
    }["Pt.useMemo"], [
        a,
        s,
        n
    ]), i = b(o, void 0), d = b(u, !1);
    return {
        message: i,
        send: c,
        isSending: d
    };
}
function kt(e) {
    const t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["f"])(), n = O(t);
    return {
        buttonProps: (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
            "kt.useMemo": ()=>{
                const { className: a, disconnect: c } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["h"])(t);
                return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["m"])(e, {
                    className: a,
                    onClick: {
                        "kt.useMemo": ()=>c(e.stopTracks ?? !0)
                    }["kt.useMemo"],
                    disabled: n === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ConnectionState"].Disconnected
                });
            }
        }["kt.useMemo"], [
            t,
            e,
            n
        ])
    };
}
function Ye(e) {
    if (e.publication instanceof __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["LocalTrackPublication"]) {
        const t = e.publication.track;
        if (t) {
            const { facingMode: n } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["facingModeFromLocalTrack"])(t);
            return n;
        }
    }
    return "undefined";
}
function Tt({ trackRef: e, props: t }) {
    const n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["i"])(e), s = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["j"])(), { className: a } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Tt.useMemo": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["k"])()
    }["Tt.useMemo"], []), c = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Tt.useMemo[c]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["m"])(n, s == null ? void 0 : s.pin.state)
    }["Tt.useMemo[c]"], [
        n,
        s == null ? void 0 : s.pin.state
    ]);
    return {
        mergedProps: (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
            "Tt.useMemo": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["m"])(t, {
                    className: a,
                    onClick: {
                        "Tt.useMemo": (u)=>{
                            var i, d, f, l, p;
                            (i = t.onClick) == null || i.call(t, u), c ? (f = s == null ? void 0 : (d = s.pin).dispatch) == null || f.call(d, {
                                msg: "clear_pin"
                            }) : (p = s == null ? void 0 : (l = s.pin).dispatch) == null || p.call(l, {
                                msg: "set_pin",
                                trackReference: n
                            });
                        }
                    }["Tt.useMemo"]
                })
        }["Tt.useMemo"], [
            t,
            a,
            n,
            c,
            s == null ? void 0 : s.pin
        ]),
        inFocus: c
    };
}
function Et(e, t, n = {}) {
    const s = n.gridLayouts ?? __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["G"], { width: a, height: c } = Xe(e), o = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["n"])(s, t, a, c);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Et.useEffect": ()=>{
            e.current && o && (e.current.style.setProperty("--lk-col-count", o == null ? void 0 : o.columns.toString()), e.current.style.setProperty("--lk-row-count", o == null ? void 0 : o.rows.toString()));
        }
    }["Et.useEffect"], [
        e,
        o
    ]), {
        layout: o,
        containerWidth: a,
        containerHeight: c
    };
}
function q(e, t = {}) {
    var u, i;
    const n = typeof e == "string" ? t.participant : e.participant, s = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["c"])(n), a = typeof e == "string" ? {
        participant: s,
        source: e
    } : e, [c, o] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(!!((u = a.publication) != null && u.isMuted || (i = s.getTrackPublication(a.source)) != null && i.isMuted));
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "q.useEffect": ()=>{
            const d = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["o"])(a).subscribe(o);
            return ({
                "q.useEffect": ()=>d.unsubscribe()
            })["q.useEffect"];
        }
    }["q.useEffect"], [
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["p"])(a)
    ]), c;
}
function Ze(e) {
    const t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["c"])(e), n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Ze.useMemo[n]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["q"])(t)
    }["Ze.useMemo[n]"], [
        t
    ]);
    return b(n, t.isSpeaking);
}
function Ke(e = {}) {
    const t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["u"])(e.room), [n, s] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(t.localParticipant), [a, c] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(n.isMicrophoneEnabled), [o, u] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(n.isCameraEnabled), [i, d] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(n.isScreenShareEnabled), [f, l] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(n.lastMicrophoneError), [p, S] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(n.lastCameraError), [g, m] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(void 0), [y, h] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(void 0), P = (M)=>{
        u(M.isCameraEnabled), c(M.isMicrophoneEnabled), d(M.isScreenShareEnabled), h(M.cameraTrack), m(M.microphoneTrack), l(M.participant.lastMicrophoneError), S(M.participant.lastCameraError), s(M.participant);
    };
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Ke.useEffect": ()=>{
            const M = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["t"])(t.localParticipant).subscribe(P);
            return ({
                "Ke.useEffect": ()=>M.unsubscribe()
            })["Ke.useEffect"];
        }
    }["Ke.useEffect"], [
        t
    ]), {
        isMicrophoneEnabled: a,
        isScreenShareEnabled: i,
        isCameraEnabled: o,
        microphoneTrack: g,
        cameraTrack: y,
        lastMicrophoneError: f,
        lastCameraError: p,
        localParticipant: n
    };
}
function wt() {
    const e = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["f"])(), t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "wt.useMemo[t]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["v"])(e.localParticipant)
    }["wt.useMemo[t]"], [
        e
    ]);
    return b(t, e.localParticipant.permissions);
}
function At({ kind: e, room: t, track: n, requestPermissions: s, onError: a }) {
    const c = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["w"])(), o = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "At.useMemo[o]": ()=>t ?? c ?? new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Room"]()
    }["At.useMemo[o]"], [
        t,
        c
    ]), u = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "At.useMemo[u]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["x"])(e, a, s)
    }["At.useMemo[u]"], [
        e,
        s,
        a
    ]), i = b(u, []), [d, f] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])((o == null ? void 0 : o.getActiveDevice(e)) ?? "default"), { className: l, activeDeviceObservable: p, setActiveMediaDevice: S } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "At.useMemo": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["y"])(e, o)
    }["At.useMemo"], [
        e,
        o,
        n
    ]);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "At.useEffect": ()=>{
            const g = p.subscribe({
                "At.useEffect.g": (m)=>{
                    m && (__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].info("setCurrentDeviceId", m), f(m));
                }
            }["At.useEffect.g"]);
            return ({
                "At.useEffect": ()=>{
                    g == null || g.unsubscribe();
                }
            })["At.useEffect"];
        }
    }["At.useEffect"], [
        p
    ]), {
        devices: i,
        className: l,
        activeDeviceId: d,
        setActiveMediaDevice: S
    };
}
function Ct({ kind: e, onError: t }) {
    const n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Ct.useMemo[n]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["x"])(e, t)
    }["Ct.useMemo[n]"], [
        e,
        t
    ]);
    return b(n, []);
}
function et(e, t, n = {}) {
    const s = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])([]), a = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(-1), c = t !== a.current, o = typeof n.customSortFunction == "function" ? n.customSortFunction(e) : (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["z"])(e);
    let u = [
        ...o
    ];
    if (c === !1) try {
        u = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["A"])(s.current, o, t);
    } catch (i) {
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].error("Error while running updatePages(): ", i);
    }
    return c ? s.current = o : s.current = u, a.current = t, u;
}
function Ot(e, t) {
    const [n, s] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(1), a = Math.max(Math.ceil(t.length / e), 1);
    n > a && s(a);
    const c = n * e, o = c - e, u = (l)=>{
        s((p)=>l === "next" ? p === a ? p : p + 1 : p === 1 ? p : p - 1);
    }, i = (l)=>{
        l > a ? s(a) : l < 1 ? s(1) : s(l);
    }, f = et(t, e).slice(o, c);
    return {
        totalPageCount: a,
        nextPage: ()=>u("next"),
        prevPage: ()=>u("previous"),
        setPage: i,
        firstItemIndex: o,
        lastItemIndex: c,
        tracks: f,
        currentPage: n
    };
}
function It(e = {}) {
    let t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["B"])();
    e.participant && (t = e.participant);
    const n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "It.useMemo[n]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["C"])(t)
    }["It.useMemo[n]"], [
        t
    ]), { identity: s, name: a, metadata: c } = b(n, {
        name: t == null ? void 0 : t.name,
        identity: t == null ? void 0 : t.identity,
        metadata: t == null ? void 0 : t.metadata
    });
    return {
        identity: s,
        name: a,
        metadata: c
    };
}
function Dt(e = {}) {
    const t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["c"])(e.participant), n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Dt.useMemo[n]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["v"])(t)
    }["Dt.useMemo[n]"], [
        t
    ]);
    return b(n, t.permissions);
}
function Lt({ trackRef: e, onParticipantClick: t, disableSpeakingIndicator: n, htmlProps: s }) {
    const a = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["i"])(e), c = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Lt.useMemo[c]": ()=>{
            const { className: p } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["D"])();
            return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["m"])(s, {
                className: p,
                onClick: {
                    "Lt.useMemo[c]": (S)=>{
                        var g;
                        if ((g = s.onClick) == null || g.call(s, S), typeof t == "function") {
                            const m = a.publication ?? a.participant.getTrackPublication(a.source);
                            t({
                                participant: a.participant,
                                track: m
                            });
                        }
                    }
                }["Lt.useMemo[c]"]
            });
        }
    }["Lt.useMemo[c]"], [
        s,
        t,
        a.publication,
        a.source,
        a.participant
    ]), o = a.participant.getTrackPublication(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Microphone), u = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Lt.useMemo[u]": ()=>({
                participant: a.participant,
                source: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Microphone,
                publication: o
            })
    }["Lt.useMemo[u]"], [
        o,
        a.participant
    ]), i = q(a), d = q(u), f = Ze(a.participant), l = Ye(a);
    return {
        elementProps: {
            "data-lk-audio-muted": d,
            "data-lk-video-muted": i,
            "data-lk-speaking": n === !0 ? !1 : f,
            "data-lk-local-participant": a.participant.isLocal,
            "data-lk-source": a.source,
            "data-lk-facing-mode": l,
            ...c
        }
    };
}
function _(e = {}) {
    const t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["u"])(e.room), [n, s] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])([]);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "_.useEffect": ()=>{
            const a = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["E"])(t, {
                additionalRoomEvents: e.updateOnlyOn
            }).subscribe(s);
            return ({
                "_.useEffect": ()=>a.unsubscribe()
            })["_.useEffect"];
        }
    }["_.useEffect"], [
        t,
        JSON.stringify(e.updateOnlyOn)
    ]), n;
}
function Rt(e = {}) {
    const t = _(e), { localParticipant: n } = Ke(e);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Rt.useMemo": ()=>[
                n,
                ...t
            ]
    }["Rt.useMemo"], [
        n,
        t
    ]);
}
function Nt(e) {
    return e = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["F"])(e), (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Nt.useMemo": ()=>(e == null ? void 0 : e.pin.state) !== void 0 && e.pin.state.length >= 1 ? e.pin.state : []
    }["Nt.useMemo"], [
        e.pin.state
    ]);
}
function Ft(e, t = {}) {
    const n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["f"])(), [s] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(t.updateOnlyOn), a = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Ft.useMemo[a]": ()=>typeof e == "string" ? (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["H"])(n, e, {
                additionalEvents: s
            }) : (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["I"])(n, e, {
                additionalEvents: s
            })
    }["Ft.useMemo[a]"], [
        n,
        JSON.stringify(e),
        s
    ]), [c, o] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])({
        p: void 0
    });
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Ft.useEffect": ()=>{
            const u = a.subscribe({
                "Ft.useEffect.u": (i)=>o({
                        p: i
                    })
            }["Ft.useEffect.u"]);
            return ({
                "Ft.useEffect": ()=>u.unsubscribe()
            })["Ft.useEffect"];
        }
    }["Ft.useEffect"], [
        a
    ]), c.p;
}
function Vt(e = {}) {
    const t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["u"])(e.room), n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Vt.useMemo[n]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["J"])(t)
    }["Vt.useMemo[n]"], [
        t
    ]), { name: s, metadata: a } = b(n, {
        name: t.name,
        metadata: t.metadata
    });
    return {
        name: s,
        metadata: a
    };
}
function tt() {
    const e = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["f"])(), t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "tt.useMemo[t]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["K"])(e)
    }["tt.useMemo[t]"], [
        e
    ]);
    return b(t, e.activeSpeakers);
}
function _t(e) {
    const [t, n] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])((0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["M"])(e)), s = tt();
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "_t.useEffect": ()=>{
            n((0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["M"])(e));
        }
    }["_t.useEffect"], [
        s,
        e
    ]), t;
}
function zt({ room: e, props: t }) {
    const n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["u"])(e), { className: s, roomAudioPlaybackAllowedObservable: a, handleStartAudioPlayback: c } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "zt.useMemo": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["N"])()
    }["zt.useMemo"], []), o = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "zt.useMemo[o]": ()=>a(n)
    }["zt.useMemo[o]"], [
        n,
        a
    ]), { canPlayAudio: u } = b(o, {
        canPlayAudio: n.canPlaybackAudio
    });
    return {
        mergedProps: (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
            "zt.useMemo": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["m"])(t, {
                    className: s,
                    onClick: {
                        "zt.useMemo": ()=>{
                            c(n);
                        }
                    }["zt.useMemo"],
                    style: {
                        display: u ? "none" : "block"
                    }
                })
        }["zt.useMemo"], [
            t,
            s,
            u,
            c,
            n
        ]),
        canPlayAudio: u
    };
}
function Bt({ room: e, props: t }) {
    const n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["u"])(e), { className: s, roomVideoPlaybackAllowedObservable: a, handleStartVideoPlayback: c } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Bt.useMemo": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["O"])()
    }["Bt.useMemo"], []), o = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Bt.useMemo[o]": ()=>a(n)
    }["Bt.useMemo[o]"], [
        n,
        a
    ]), { canPlayVideo: u } = b(o, {
        canPlayVideo: n.canPlaybackVideo
    });
    return {
        mergedProps: (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
            "Bt.useMemo": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["m"])(t, {
                    className: s,
                    onClick: {
                        "Bt.useMemo": ()=>{
                            c(n);
                        }
                    }["Bt.useMemo"],
                    style: {
                        display: u ? "none" : "block"
                    }
                })
        }["Bt.useMemo"], [
            t,
            s,
            u,
            c,
            n
        ]),
        canPlayVideo: u
    };
}
function qt(e, t = {}) {
    const n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null), s = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null), a = t.minSwipeDistance ?? 50, c = (i)=>{
        s.current = null, n.current = i.targetTouches[0].clientX;
    }, o = (i)=>{
        s.current = i.targetTouches[0].clientX;
    }, u = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "qt.useCallback[u]": ()=>{
            if (!n.current || !s.current) return;
            const i = n.current - s.current, d = i > a, f = i < -a;
            d && t.onLeftSwipe && t.onLeftSwipe(), f && t.onRightSwipe && t.onRightSwipe();
        }
    }["qt.useCallback[u]"], [
        a,
        t
    ]);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "qt.useEffect": ()=>{
            const i = e.current;
            return i && (i.addEventListener("touchstart", c, {
                passive: !0
            }), i.addEventListener("touchmove", o, {
                passive: !0
            }), i.addEventListener("touchend", u, {
                passive: !0
            })), ({
                "qt.useEffect": ()=>{
                    i && (i.removeEventListener("touchstart", c), i.removeEventListener("touchmove", o), i.removeEventListener("touchend", u));
                }
            })["qt.useEffect"];
        }
    }["qt.useEffect"], [
        e,
        u
    ]);
}
function Ut({ props: e }) {
    const { dispatch: t, state: n } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a"])().widget, { className: s } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Ut.useMemo": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["P"])()
    }["Ut.useMemo"], []);
    return {
        mergedProps: (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
            "Ut.useMemo": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["m"])(e, {
                    className: s,
                    onClick: {
                        "Ut.useMemo": ()=>{
                            t && t({
                                msg: "toggle_chat"
                            });
                        }
                    }["Ut.useMemo"],
                    "aria-pressed": n != null && n.showChat ? "true" : "false",
                    "data-lk-unread-msgs": n ? n.unreadMessages < 10 ? n.unreadMessages.toFixed(0) : "9+" : "0"
                })
        }["Ut.useMemo"], [
            e,
            s,
            t,
            n
        ])
    };
}
function Jt(e, t, n = {}) {
    const [s, a] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(void 0);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Jt.useEffect": ()=>{
            var o;
            if (e === void 0) throw Error("token endpoint needs to be defined");
            if (((o = n.userInfo) == null ? void 0 : o.identity) === void 0) return;
            ({
                "Jt.useEffect": async ()=>{
                    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].debug("fetching token");
                    const u = new URLSearchParams({
                        ...n.userInfo,
                        roomName: t
                    }), i = await fetch(`${e}?${u.toString()}`);
                    if (!i.ok) {
                        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].error(`Could not fetch token. Server responded with status ${i.status}: ${i.statusText}`);
                        return;
                    }
                    const { accessToken: d } = await i.json();
                    a(d);
                }
            })["Jt.useEffect"]();
        }
    }["Jt.useEffect"], [
        e,
        t,
        JSON.stringify(n)
    ]), s;
}
function xt(e) {
    var c, o;
    const t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["i"])(e), { className: n, mediaMutedObserver: s } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "xt.useMemo": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Q"])(t)
    }["xt.useMemo"], [
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["p"])(t)
    ]);
    return {
        isMuted: b(s, !!((c = t.publication) != null && c.isMuted || (o = t.participant.getTrackPublication(t.source)) != null && o.isMuted)),
        className: n
    };
}
function Wt({ source: e, onChange: t, initialState: n, captureOptions: s, publishOptions: a, onDeviceError: c, ...o }) {
    var P;
    const u = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["w"])(), i = (P = u == null ? void 0 : u.localParticipant) == null ? void 0 : P.getTrackPublication(e), d = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(!1), { toggle: f, className: l, pendingObserver: p, enabledObserver: S } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Wt.useMemo": ()=>u ? (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["S"])(e, u, s, a, c) : (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["T"])()
    }["Wt.useMemo"], [
        u,
        e,
        JSON.stringify(s),
        a
    ]), g = b(p, !1), m = b(S, n ?? !!(i != null && i.isEnabled));
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Wt.useEffect": ()=>{
            t == null || t(m, d.current), d.current = !1;
        }
    }["Wt.useEffect"], [
        m,
        t
    ]), (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Wt.useEffect": ()=>{
            n !== void 0 && (__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].debug("forcing initial toggle state", e, n), f(n));
        }
    }["Wt.useEffect"], []);
    const y = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Wt.useMemo[y]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["m"])(o, {
                className: l
            })
    }["Wt.useMemo[y]"], [
        o,
        l
    ]), h = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "Wt.useCallback[h]": (M)=>{
            var I;
            d.current = !0, f().catch({
                "Wt.useCallback[h]": ()=>d.current = !1
            }["Wt.useCallback[h]"]), (I = o.onClick) == null || I.call(o, M);
        }
    }["Wt.useCallback[h]"], [
        o,
        f
    ]);
    return {
        toggle: f,
        enabled: m,
        pending: g,
        track: i,
        buttonProps: {
            ...y,
            "aria-pressed": m,
            "data-lk-source": e,
            "data-lk-enabled": m,
            disabled: g,
            onClick: h
        }
    };
}
function Gt(e = [
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Camera,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Microphone,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.ScreenShare,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.ScreenShareAudio,
    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Unknown
], t = {}) {
    const n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["u"])(t.room), [s, a] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])([]), [c, o] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])([]), u = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Gt.useMemo[u]": ()=>e.map({
                "Gt.useMemo[u]": (d)=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["U"])(d) ? d.source : d
            }["Gt.useMemo[u]"])
    }["Gt.useMemo[u]"], [
        JSON.stringify(e)
    ]);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Gt.useEffect": ()=>{
            const d = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["V"])(n, u, {
                additionalRoomEvents: t.updateOnlyOn,
                onlySubscribed: t.onlySubscribed
            }).subscribe({
                "Gt.useEffect.d": ({ trackReferences: f, participants: l })=>{
                    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].debug("setting track bundles", f, l), a(f), o(l);
                }
            }["Gt.useEffect.d"]);
            return ({
                "Gt.useEffect": ()=>d.unsubscribe()
            })["Gt.useEffect"];
        }
    }["Gt.useEffect"], [
        n,
        JSON.stringify(t.onlySubscribed),
        JSON.stringify(t.updateOnlyOn),
        JSON.stringify(e)
    ]), (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Gt.useMemo": ()=>{
            if ((0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["W"])(e)) {
                const d = st(e, c), f = Array.from(s);
                return c.forEach({
                    "Gt.useMemo": (l)=>{
                        d.has(l.identity) && (d.get(l.identity) ?? []).forEach({
                            "Gt.useMemo": (S)=>{
                                if (s.find({
                                    "Gt.useMemo": ({ participant: m, publication: y })=>l.identity === m.identity && y.source === S
                                }["Gt.useMemo"])) return;
                                __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].debug(`Add ${S} placeholder for participant ${l.identity}.`);
                                const g = {
                                    participant: l,
                                    source: S
                                };
                                f.push(g);
                            }
                        }["Gt.useMemo"]);
                    }
                }["Gt.useMemo"]), f;
            } else return s;
        }
    }["Gt.useMemo"], [
        s,
        c,
        e
    ]);
}
function nt(e, t) {
    const n = new Set(e);
    for (const s of t)n.delete(s);
    return n;
}
function st(e, t) {
    const n = /* @__PURE__ */ new Map();
    if ((0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["W"])(e)) {
        const s = e.filter((a)=>a.withPlaceholder).map((a)=>a.source);
        t.forEach((a)=>{
            const c = a.getTrackPublications().map((u)=>{
                var i;
                return (i = u.track) == null ? void 0 : i.source;
            }).filter((u)=>u !== void 0), o = Array.from(nt(new Set(s), new Set(c)));
            o.length > 0 && n.set(a.identity, o);
        });
    }
    return n;
}
function at(e) {
    const [t, n] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])((0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["X"])(e)), { trackObserver: s } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "at.useMemo": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Y"])(e)
    }["at.useMemo"], [
        e.participant.sid ?? e.participant.identity,
        e.source
    ]);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "at.useEffect": ()=>{
            const a = s.subscribe({
                "at.useEffect.a": (c)=>{
                    n(c);
                }
            }["at.useEffect.a"]);
            return ({
                "at.useEffect": ()=>a == null ? void 0 : a.unsubscribe()
            })["at.useEffect"];
        }
    }["at.useEffect"], [
        s
    ]), {
        participant: e.participant,
        source: e.source ?? __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Unknown,
        publication: t
    };
}
function Qt(e, t) {
    const n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["c"])(t);
    return at({
        name: e,
        participant: n
    });
}
function $t(e) {
    const t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["f"])(), n = O(t), s = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "$t.useMemo[s]": ()=>n === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ConnectionState"].Disconnected
    }["$t.useMemo[s]"], [
        n
    ]), a = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "$t.useMemo[a]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Z"])(t, e)
    }["$t.useMemo[a]"], [
        t,
        e,
        s
    ]), c = b(a.isSendingObservable, !1), o = b(a.messageObservable, []);
    return {
        send: a.send,
        chatMessages: o,
        isSending: c
    };
}
function Ht(e = {}) {
    const [t, n] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])((0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["_"])(e.defaults, e.preventLoad ?? !1)), s = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "Ht.useCallback[s]": (i)=>{
            n({
                "Ht.useCallback[s]": (d)=>({
                        ...d,
                        audioEnabled: i
                    })
            }["Ht.useCallback[s]"]);
        }
    }["Ht.useCallback[s]"], []), a = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "Ht.useCallback[a]": (i)=>{
            n({
                "Ht.useCallback[a]": (d)=>({
                        ...d,
                        videoEnabled: i
                    })
            }["Ht.useCallback[a]"]);
        }
    }["Ht.useCallback[a]"], []), c = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "Ht.useCallback[c]": (i)=>{
            n({
                "Ht.useCallback[c]": (d)=>({
                        ...d,
                        audioDeviceId: i
                    })
            }["Ht.useCallback[c]"]);
        }
    }["Ht.useCallback[c]"], []), o = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "Ht.useCallback[o]": (i)=>{
            n({
                "Ht.useCallback[o]": (d)=>({
                        ...d,
                        videoDeviceId: i
                    })
            }["Ht.useCallback[o]"]);
        }
    }["Ht.useCallback[o]"], []), u = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "Ht.useCallback[u]": (i)=>{
            n({
                "Ht.useCallback[u]": (d)=>({
                        ...d,
                        username: i
                    })
            }["Ht.useCallback[u]"]);
        }
    }["Ht.useCallback[u]"], []);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Ht.useEffect": ()=>{
            (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["$"])(t, e.preventSave ?? !1);
        }
    }["Ht.useEffect"], [
        t,
        e.preventSave
    ]), {
        userChoices: t,
        saveAudioInputEnabled: s,
        saveVideoInputEnabled: a,
        saveAudioInputDeviceId: c,
        saveVideoInputDeviceId: o,
        saveUsername: u
    };
}
function jt(e, t = {}) {
    const n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["c"])(e), s = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["u"])(t.room), a = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "jt.useMemo[a]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a0"])(s, n)
    }["jt.useMemo[a]"], [
        s,
        n
    ]);
    return b(a, n.isLocal ? n.isE2EEEnabled : !!(n != null && n.isEncrypted));
}
function Xt(e, t = {
    fftSize: 32,
    smoothingTimeConstant: 0
}) {
    const n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a1"])(e) ? e.publication.track : e, [s, a] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(0);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Xt.useEffect": ()=>{
            if (!n || !n.mediaStream) return;
            const { cleanup: c, analyser: o } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createAudioAnalyser"])(n, t), u = o.frequencyBinCount, i = new Uint8Array(u), f = setInterval({
                "Xt.useEffect.f": ()=>{
                    o.getByteFrequencyData(i);
                    let l = 0;
                    for(let p = 0; p < i.length; p++){
                        const S = i[p];
                        l += S * S;
                    }
                    a(Math.sqrt(l / i.length) / 255);
                }
            }["Xt.useEffect.f"], 1e3 / 30);
            return ({
                "Xt.useEffect": ()=>{
                    c(), clearInterval(f);
                }
            })["Xt.useEffect"];
        }
    }["Xt.useEffect"], [
        n,
        n == null ? void 0 : n.mediaStream,
        JSON.stringify(t)
    ]), s;
}
const rt = (e)=>{
    const t = (n)=>{
        let c = 1 - Math.max(-100, Math.min(-10, n)) * -1 / 100;
        return c = Math.sqrt(c), c;
    };
    return e.map((n)=>n === -1 / 0 ? 0 : t(n));
}, ct = {
    bands: 5,
    loPass: 100,
    hiPass: 600,
    updateInterval: 32,
    analyserOptions: {
        fftSize: 2048
    }
};
function Yt(e, t = {}) {
    var o;
    const n = e instanceof __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"] ? e : (o = e == null ? void 0 : e.publication) == null ? void 0 : o.track, s = {
        ...ct,
        ...t
    }, [a, c] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(new Array(s.bands).fill(0));
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Yt.useEffect": ()=>{
            if (!n || !(n != null && n.mediaStream)) return;
            const { analyser: u, cleanup: i } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createAudioAnalyser"])(n, s.analyserOptions), d = u.frequencyBinCount, f = new Float32Array(d), p = setInterval({
                "Yt.useEffect.p": ()=>{
                    u.getFloatFrequencyData(f);
                    let S = new Float32Array(f.length);
                    for(let h = 0; h < f.length; h++)S[h] = f[h];
                    S = S.slice(t.loPass, t.hiPass);
                    const g = rt(S), m = Math.ceil(g.length / s.bands), y = [];
                    for(let h = 0; h < s.bands; h++){
                        const P = g.slice(h * m, (h + 1) * m).reduce({
                            "Yt.useEffect.p.P": (M, I)=>M += I
                        }["Yt.useEffect.p.P"], 0);
                        y.push(P / m);
                    }
                    c(y);
                }
            }["Yt.useEffect.p"], s.updateInterval);
            return ({
                "Yt.useEffect": ()=>{
                    i(), clearInterval(p);
                }
            })["Yt.useEffect"];
        }
    }["Yt.useEffect"], [
        n,
        n == null ? void 0 : n.mediaStream,
        JSON.stringify(t)
    ]), a;
}
const ot = {
    barCount: 120,
    volMultiplier: 5,
    updateInterval: 20
};
function Zt(e, t = {}) {
    var f;
    const n = e instanceof __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"] ? e : (f = e == null ? void 0 : e.publication) == null ? void 0 : f.track, s = {
        ...ot,
        ...t
    }, a = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(new Float32Array()), c = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(performance.now()), o = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(0), [u, i] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])([]), d = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "Zt.useCallback[d]": (l)=>{
            i(Array.from(ut(l, s.barCount).map({
                "Zt.useCallback[d]": (p)=>Math.sqrt(p) * s.volMultiplier
            }["Zt.useCallback[d]"])));
        }
    }["Zt.useCallback[d]"], []);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Zt.useEffect": ()=>{
            if (!n || !(n != null && n.mediaStream)) return;
            const { analyser: l, cleanup: p } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createAudioAnalyser"])(n, {
                fftSize: U(s.barCount)
            }), S = U(s.barCount), g = new Float32Array(S), m = {
                "Zt.useEffect.m": ()=>{
                    if (y = requestAnimationFrame(m), l.getFloatTimeDomainData(g), a.current.map({
                        "Zt.useEffect.m": (h, P)=>h + g[P]
                    }["Zt.useEffect.m"]), o.current += 1, performance.now() - c.current >= s.updateInterval) {
                        const h = g.map({
                            "Zt.useEffect.m.h": (P)=>P / o.current
                        }["Zt.useEffect.m.h"]);
                        d(h), c.current = performance.now(), o.current = 0;
                    }
                }
            }["Zt.useEffect.m"];
            let y = requestAnimationFrame(m);
            return ({
                "Zt.useEffect": ()=>{
                    p(), cancelAnimationFrame(y);
                }
            })["Zt.useEffect"];
        }
    }["Zt.useEffect"], [
        n,
        n == null ? void 0 : n.mediaStream,
        JSON.stringify(t),
        d
    ]), {
        bars: u
    };
}
function U(e) {
    return e < 32 ? 32 : it(e);
}
function it(e) {
    let t = 2;
    for(; e >>= 1;)t <<= 1;
    return t;
}
function ut(e, t) {
    const n = Math.floor(e.length / t), s = new Float32Array(t);
    for(let a = 0; a < t; a++){
        const c = n * a;
        let o = 0;
        for(let u = 0; u < n; u++)o = o + Math.abs(e[c + u]);
        s[a] = o / n;
    }
    return s;
}
function J(e, t) {
    const n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["B"])(), s = _({
        updateOnlyOn: []
    }), a = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "J.useMemo[a]": ()=>t ? s.find({
                "J.useMemo[a]": (u)=>u.identity === t
            }["J.useMemo[a]"]) : n
    }["J.useMemo[a]"], [
        t,
        s,
        n
    ]), c = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "J.useMemo[c]": ()=>{
            if (a) return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a2"])(a, {
                sources: e
            });
        }
    }["J.useMemo[c]"], [
        a,
        JSON.stringify(e)
    ]);
    return b(c, []);
}
function dt(e) {
    var n, s, a;
    const t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "dt.useMemo[t]": ()=>{
            var c;
            return (c = e == null ? void 0 : e.publication) != null && c.track ? (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a3"])(e == null ? void 0 : e.publication.track) : void 0;
        }
    }["dt.useMemo[t]"], [
        (n = e == null ? void 0 : e.publication) == null ? void 0 : n.track
    ]);
    return b(t, {
        timestamp: Date.now(),
        rtpTimestamp: (a = (s = e == null ? void 0 : e.publication) == null ? void 0 : s.track) == null ? void 0 : a.rtpTimestamp
    });
}
const lt = {
    bufferSize: 100
};
function ft(e, t) {
    const n = {
        ...lt,
        ...t
    }, [s, a] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])([]), c = dt(e), o = (u)=>{
        var i;
        (i = n.onTranscription) == null || i.call(n, u), a((d)=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a5"])(d, // when first receiving a segment, add the current media timestamp to it
            u.map((f)=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a6"])(f, c)), n.bufferSize));
    };
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "ft.useEffect": ()=>{
            if (!(e != null && e.publication)) return;
            const u = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a4"])(e.publication).subscribe({
                "ft.useEffect.u": (i)=>{
                    o(...i);
                }
            }["ft.useEffect.u"]);
            return ({
                "ft.useEffect": ()=>{
                    u.unsubscribe();
                }
            })["ft.useEffect"];
        }
    }["ft.useEffect"], [
        e && (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["p"])(e),
        o
    ]), {
        segments: s
    };
}
function pt(e = {}) {
    const t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["B"])(), n = e.participant ?? t, s = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "pt.useMemo[s]": // weird typescript constraint
        ()=>n ? (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a7"])(n) : (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a7"])(n)
    }["pt.useMemo[s]"], [
        n
    ]);
    return b(s, {
        attributes: n == null ? void 0 : n.attributes
    });
}
function Kt(e, t = {}) {
    const n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["c"])(t.participant), [s, a] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(n.attributes[e]);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Kt.useEffect": ()=>{
            if (!n) return;
            const c = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a7"])(n).subscribe({
                "Kt.useEffect.c": (o)=>{
                    o.changed[e] !== void 0 && a(o.attributes[e]);
                }
            }["Kt.useEffect.c"]);
            return ({
                "Kt.useEffect": ()=>{
                    c.unsubscribe();
                }
            })["Kt.useEffect"];
        }
    }["Kt.useEffect"], [
        n,
        e
    ]), s;
}
const x = "lk.agent.state";
function en() {
    const e = _(), t = e.find((l)=>l.kind === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantKind"].AGENT && !("lk.publish_on_behalf" in l.attributes)), n = e.find((l)=>l.kind === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ParticipantKind"].AGENT && l.attributes["lk.publish_on_behalf"] === (t == null ? void 0 : t.identity)), s = J([
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Microphone,
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Camera
    ], t == null ? void 0 : t.identity), a = J([
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Microphone,
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Camera
    ], n == null ? void 0 : n.identity), c = s.find((l)=>l.source === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Microphone) ?? a.find((l)=>l.source === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Microphone), o = s.find((l)=>l.source === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Camera) ?? a.find((l)=>l.source === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Camera), { segments: u } = ft(c), i = O(), { attributes: d } = pt({
        participant: t
    }), f = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "en.useMemo[f]": ()=>i === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ConnectionState"].Disconnected ? "disconnected" : i === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ConnectionState"].Connecting || !t || !(d != null && d[x]) ? "connecting" : d[x]
    }["en.useMemo[f]"], [
        d,
        t,
        i
    ]);
    return {
        agent: t,
        state: f,
        audioTrack: c,
        videoTrack: o,
        agentTranscriptions: u,
        agentAttributes: d
    };
}
function tn(e) {
    const t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["u"])(e), n = O(t), s = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "tn.useMemo[s]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a8"])(t)
    }["tn.useMemo[s]"], [
        t,
        n
    ]);
    return b(s, t.isRecording);
}
function bt(e) {
    const t = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["f"])(), s = O(t) === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ConnectionState"].Disconnected, a = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "bt.useMemo[a]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a9"])(t, e)
    }["bt.useMemo[a]"], [
        t,
        e
    ]);
    return {
        textStreams: b(s ? void 0 : a, [])
    };
}
function nn(e) {
    const { participantIdentities: t, trackSids: n } = e ?? {}, { textStreams: s } = bt(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["aa"].TRANSCRIPTION);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "nn.useMemo": ()=>s.filter({
                "nn.useMemo": (c)=>t ? t.includes(c.participantInfo.identity) : !0
            }["nn.useMemo"]).filter({
                "nn.useMemo": (c)=>{
                    var o;
                    return n ? n.includes(((o = c.streamInfo.attributes) == null ? void 0 : o["lk.transcribed_track_id"]) ?? "") : !0;
                }
            }["nn.useMemo"])
    }["nn.useMemo"], [
        s,
        t,
        n
    ]);
}
;
 //# sourceMappingURL=hooks-COF-7zxu.mjs.map
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/components-k0KtCs0w.mjs [app-client] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "A": (()=>wa),
    "B": (()=>Xt),
    "C": (()=>ra),
    "D": (()=>ca),
    "E": (()=>vt),
    "F": (()=>ha),
    "G": (()=>va),
    "H": (()=>ie),
    "I": (()=>Et),
    "J": (()=>pt),
    "K": (()=>wt),
    "L": (()=>pa),
    "M": (()=>ua),
    "N": (()=>kt),
    "O": (()=>Rt),
    "P": (()=>Bt),
    "Q": (()=>Mt),
    "R": (()=>Ra),
    "S": (()=>la),
    "T": (()=>fa),
    "U": (()=>yt),
    "V": (()=>Ft),
    "W": (()=>bt),
    "X": (()=>Me),
    "Y": (()=>St),
    "Z": (()=>oe),
    "_": (()=>Ct),
    "a": (()=>Sa),
    "b": (()=>Pt),
    "c": (()=>sa),
    "d": (()=>ia),
    "e": (()=>oa),
    "f": (()=>ma),
    "g": (()=>Ea),
    "h": (()=>ga),
    "i": (()=>ba),
    "j": (()=>Ie),
    "k": (()=>ya),
    "l": (()=>$t),
    "m": (()=>Ca),
    "n": (()=>aa),
    "o": (()=>na),
    "p": (()=>It),
    "q": (()=>da),
    "r": (()=>be),
    "s": (()=>ne),
    "t": (()=>K),
    "u": (()=>Se),
    "v": (()=>ka),
    "w": (()=>Ma),
    "x": (()=>Zt),
    "y": (()=>Ht),
    "z": (()=>gt)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/room-DhBnHppi.mjs [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-client@2.15.2_@types+dom-mediacapture-record@1.0.22/node_modules/livekit-client/dist/livekit-client.esm.mjs [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/contexts-CH4CA2DN.mjs [app-client] (ecmascript)");
;
;
;
;
;
;
const aa = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function(n, a) {
    const { buttonProps: r } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["u"])(n);
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("button", {
        ref: a,
        ...r
    }, n.children);
}), na = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function({ room: n, ...a }, r) {
    const c = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a"])(n);
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("div", {
        ref: r,
        ...a
    }, c);
}), ra = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function(n, a) {
    const { mergedProps: r } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["b"])({
        props: n
    });
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("button", {
        ref: a,
        ...r
    }, n.children);
}), ca = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function(n, a) {
    const { buttonProps: r } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["c"])(n);
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("button", {
        ref: a,
        ...r
    }, n.children);
}), gt = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 16,
        height: 16,
        fill: "currentColor",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M1.354.646a.5.5 0 1 0-.708.708l14 14a.5.5 0 0 0 .708-.708L11 10.293V4.5A1.5 1.5 0 0 0 9.5 3H3.707zM0 4.5a1.5 1.5 0 0 1 .943-1.393l9.532 9.533c-.262.224-.603.36-.975.36h-8A1.5 1.5 0 0 1 0 11.5z"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "m15.2 3.6-2.8 2.1a1 1 0 0 0-.4.8v3a1 1 0 0 0 .4.8l2.8 2.1a.5.5 0 0 0 .8-.4V4a.5.5 0 0 0-.8-.4z"
    })), vt = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 16,
        height: 16,
        fill: "currentColor",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M0 4.5A1.5 1.5 0 0 1 1.5 3h8A1.5 1.5 0 0 1 11 4.5v7A1.5 1.5 0 0 1 9.5 13h-8A1.5 1.5 0 0 1 0 11.5zM15.2 3.6l-2.8 2.1a1 1 0 0 0-.4.8v3a1 1 0 0 0 .4.8l2.8 2.1a.5.5 0 0 0 .8-.4V4a.5.5 0 0 0-.8-.4z"
    })), la = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 16,
        height: 16,
        viewBox: "0 0 24 24",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "#FFF",
        d: "M4.99 3.99a1 1 0 0 0-.697 1.717L10.586 12l-6.293 6.293a1 1 0 1 0 1.414 1.414L12 13.414l6.293 6.293a1 1 0 1 0 1.414-1.414L13.414 12l6.293-6.293a1 1 0 0 0-.727-1.717 1 1 0 0 0-.687.303L12 10.586 5.707 4.293a1 1 0 0 0-.717-.303z"
    })), sa = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 16,
        height: 18,
        fill: "none",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M0 2.75A2.75 2.75 0 0 1 2.75 0h10.5A2.75 2.75 0 0 1 16 2.75v13.594a.75.75 0 0 1-1.234.572l-3.691-3.12a1.25 1.25 0 0 0-.807-.296H2.75A2.75 2.75 0 0 1 0 10.75v-8ZM2.75 1.5c-.69 0-1.25.56-1.25 1.25v8c0 .69.56 1.25 1.25 1.25h7.518c.65 0 1.279.23 1.775.65l2.457 2.077V2.75c0-.69-.56-1.25-1.25-1.25H2.75Z",
        clipRule: "evenodd"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M3 4.5a.5.5 0 0 1 .5-.5h9a.5.5 0 0 1 0 1h-9a.5.5 0 0 1-.5-.5Zm0 2a.5.5 0 0 1 .5-.5h9a.5.5 0 0 1 0 1h-9a.5.5 0 0 1-.5-.5Zm0 2a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5Z",
        clipRule: "evenodd"
    })), ie = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 16,
        height: 16,
        fill: "none",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentcolor",
        fillRule: "evenodd",
        d: "M5.293 2.293a1 1 0 0 1 1.414 0l4.823 4.823a1.25 1.25 0 0 1 0 1.768l-4.823 4.823a1 1 0 0 1-1.414-1.414L9.586 8 5.293 3.707a1 1 0 0 1 0-1.414z",
        clipRule: "evenodd"
    })), Et = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 16,
        height: 16,
        fill: "none",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("g", {
        stroke: "currentColor",
        strokeLinecap: "round",
        strokeLinejoin: "round",
        strokeWidth: 1.5
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M10 1.75h4.25m0 0V6m0-4.25L9 7M6 14.25H1.75m0 0V10m0 4.25L7 9"
    }))), ia = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 16,
        height: 16,
        fill: "none",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentcolor",
        fillRule: "evenodd",
        d: "M8.961.894C8.875-.298 7.125-.298 7.04.894c-.066.912-1.246 1.228-1.76.472-.67-.99-2.186-.115-1.664.96.399.824-.465 1.688-1.288 1.289-1.076-.522-1.95.994-.961 1.665.756.513.44 1.693-.472 1.759-1.192.086-1.192 1.836 0 1.922.912.066 1.228 1.246.472 1.76-.99.67-.115 2.186.96 1.664.824-.399 1.688.465 1.289 1.288-.522 1.076.994 1.95 1.665.961.513-.756 1.693-.44 1.759.472.086 1.192 1.836 1.192 1.922 0 .066-.912 1.246-1.228 1.76-.472.67.99 2.186.115 1.664-.96-.399-.824.465-1.688 1.288-1.289 1.076.522 1.95-.994.961-1.665-.756-.513-.44-1.693.472-1.759 1.192-.086 1.192-1.836 0-1.922-.912-.066-1.228-1.246-.472-1.76.99-.67.115-2.186-.96-1.664-.824.399-1.688-.465-1.289-1.288.522-1.076-.994-1.95-1.665-.961-.513.756-1.693.44-1.759-.472ZM8 13A5 5 0 1 0 8 3a5 5 0 0 0 0 10Z",
        clipRule: "evenodd"
    })), oa = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 16,
        height: 16,
        fill: "none",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M2 2.75A2.75 2.75 0 0 1 4.75 0h6.5A2.75 2.75 0 0 1 14 2.75v10.5A2.75 2.75 0 0 1 11.25 16h-6.5A2.75 2.75 0 0 1 2 13.25v-.5a.75.75 0 0 1 1.5 0v.5c0 .69.56 1.25 1.25 1.25h6.5c.69 0 1.25-.56 1.25-1.25V2.75c0-.69-.56-1.25-1.25-1.25h-6.5c-.69 0-1.25.56-1.25 1.25v.5a.75.75 0 0 1-1.5 0v-.5Z",
        clipRule: "evenodd"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M8.78 7.47a.75.75 0 0 1 0 1.06l-2.25 2.25a.75.75 0 1 1-1.06-1.06l.97-.97H1.75a.75.75 0 0 1 0-1.5h4.69l-.97-.97a.75.75 0 0 1 1.06-1.06l2.25 2.25Z",
        clipRule: "evenodd"
    })), pt = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 16,
        height: 16,
        fill: "none",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentcolor",
        fillRule: "evenodd",
        d: "M4 6.104V4a4 4 0 1 1 8 0v2.104c1.154.326 2 1.387 2 2.646v4.5A2.75 2.75 0 0 1 11.25 16h-6.5A2.75 2.75 0 0 1 2 13.25v-4.5c0-1.259.846-2.32 2-2.646ZM5.5 4a2.5 2.5 0 0 1 5 0v2h-5V4Z",
        clipRule: "evenodd"
    })), wt = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 16,
        height: 16,
        fill: "currentColor",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M12.227 11.52a5.477 5.477 0 0 0 1.246-2.97.5.5 0 0 0-.995-.1 4.478 4.478 0 0 1-.962 2.359l-1.07-1.07C10.794 9.247 11 8.647 11 8V3a3 3 0 0 0-6 0v1.293L1.354.646a.5.5 0 1 0-.708.708l14 14a.5.5 0 0 0 .708-.708zM8 12.5c.683 0 1.33-.152 1.911-.425l.743.743c-.649.359-1.378.59-2.154.66V15h2a.5.5 0 0 1 0 1h-5a.5.5 0 0 1 0-1h2v-1.522a5.502 5.502 0 0 1-4.973-4.929.5.5 0 0 1 .995-.098A4.5 4.5 0 0 0 8 12.5z"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M8.743 10.907 5 7.164V8a3 3 0 0 0 3.743 2.907z"
    })), kt = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 16,
        height: 16,
        fill: "currentColor",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fillRule: "evenodd",
        d: "M2.975 8.002a.5.5 0 0 1 .547.449 4.5 4.5 0 0 0 8.956 0 .5.5 0 1 1 .995.098A5.502 5.502 0 0 1 8.5 13.478V15h2a.5.5 0 0 1 0 1h-5a.5.5 0 0 1 0-1h2v-1.522a5.502 5.502 0 0 1-4.973-4.929.5.5 0 0 1 .448-.547z",
        clipRule: "evenodd"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M5 3a3 3 0 1 1 6 0v5a3 3 0 0 1-6 0z"
    })), Rt = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 16,
        height: 16,
        fill: "currentcolor",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M0 11.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5zm6-5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v9a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5zm6-6a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v15a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5z"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M0 11.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5zm6-5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v9a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5zm6-6a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v15a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5z"
    })), Mt = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 16,
        height: 16,
        fill: "currentcolor",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M0 11.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5zm6-5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v9a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5z"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M0 11.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5zm6-5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v9a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5z"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("g", {
        opacity: 0.25
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M12 .5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v15a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5z"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M12 .5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v15a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5z"
    }))), yt = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 16,
        height: 16,
        fill: "currentcolor",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M0 11.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5z"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M0 11.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5z"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("g", {
        opacity: 0.25
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M6 6.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v9a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5z"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M6 6.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v9a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5zm6-6a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v15a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5z"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M12 .5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v15a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5z"
    }))), bt = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 16,
        height: 16,
        fill: "currentColor",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("g", {
        opacity: 0.25
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M0 11.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-4Zm6-5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v9a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-9Zm6-6a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v15a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5V.5Z"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M0 11.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-4Zm6-5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v9a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-9Zm6-6a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v15a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5V.5Z"
    }))), Me = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 20,
        height: 16,
        fill: "none",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M0 2.75A2.75 2.75 0 0 1 2.75 0h14.5A2.75 2.75 0 0 1 20 2.75v10.5A2.75 2.75 0 0 1 17.25 16H2.75A2.75 2.75 0 0 1 0 13.25V2.75ZM2.75 1.5c-.69 0-1.25.56-1.25 1.25v10.5c0 .69.56 1.25 1.25 1.25h14.5c.69 0 1.25-.56 1.25-1.25V2.75c0-.69-.56-1.25-1.25-1.25H2.75Z",
        clipRule: "evenodd"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M9.47 4.22a.75.75 0 0 1 1.06 0l2.25 2.25a.75.75 0 0 1-1.06 1.06l-.97-.97v4.69a.75.75 0 0 1-1.5 0V6.56l-.97.97a.75.75 0 0 1-1.06-1.06l2.25-2.25Z",
        clipRule: "evenodd"
    })), St = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 20,
        height: 16,
        fill: "none",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("g", {
        fill: "currentColor"
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M7.28 4.22a.75.75 0 0 0-1.06 1.06L8.94 8l-2.72 2.72a.75.75 0 1 0 1.06 1.06L10 9.06l2.72 2.72a.75.75 0 1 0 1.06-1.06L11.06 8l2.72-2.72a.75.75 0 0 0-1.06-1.06L10 6.94z"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fillRule: "evenodd",
        d: "M2.75 0A2.75 2.75 0 0 0 0 2.75v10.5A2.75 2.75 0 0 0 2.75 16h14.5A2.75 2.75 0 0 0 20 13.25V2.75A2.75 2.75 0 0 0 17.25 0zM1.5 2.75c0-.69.56-1.25 1.25-1.25h14.5c.69 0 1.25.56 1.25 1.25v10.5c0 .69-.56 1.25-1.25 1.25H2.75c-.69 0-1.25-.56-1.25-1.25z",
        clipRule: "evenodd"
    }))), oe = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 16,
        height: 16,
        fill: "none",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M8 0a.75.75 0 0 1 .75.75v2.5a.75.75 0 0 1-1.5 0V.75A.75.75 0 0 1 8 0Z",
        clipRule: "evenodd"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M8 12a.75.75 0 0 1 .75.75v2.5a.75.75 0 0 1-1.5 0v-2.5A.75.75 0 0 1 8 12Z",
        clipRule: "evenodd",
        opacity: 0.7
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M12 1.072a.75.75 0 0 1 .274 1.024l-1.25 2.165a.75.75 0 0 1-1.299-.75l1.25-2.165A.75.75 0 0 1 12 1.072Z",
        clipRule: "evenodd"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M6 11.464a.75.75 0 0 1 .274 1.025l-1.25 2.165a.75.75 0 0 1-1.299-.75l1.25-2.165A.75.75 0 0 1 6 11.464Z",
        clipRule: "evenodd",
        opacity: 0.6
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M14.928 4a.75.75 0 0 1-.274 1.025l-2.165 1.25a.75.75 0 1 1-.75-1.3l2.165-1.25A.75.75 0 0 1 14.928 4Z",
        clipRule: "evenodd"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M4.536 10a.75.75 0 0 1-.275 1.024l-2.165 1.25a.75.75 0 0 1-.75-1.298l2.165-1.25A.75.75 0 0 1 4.536 10Z",
        clipRule: "evenodd",
        opacity: 0.5
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M16 8a.75.75 0 0 1-.75.75h-2.5a.75.75 0 0 1 0-1.5h2.5A.75.75 0 0 1 16 8Z",
        clipRule: "evenodd"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M4 8a.75.75 0 0 1-.75.75H.75a.75.75 0 0 1 0-1.5h2.5A.75.75 0 0 1 4 8Z",
        clipRule: "evenodd",
        opacity: 0.4
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M14.928 12a.75.75 0 0 1-1.024.274l-2.165-1.25a.75.75 0 0 1 .75-1.299l2.165 1.25A.75.75 0 0 1 14.928 12Z",
        clipRule: "evenodd",
        opacity: 0.9
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M4.536 6a.75.75 0 0 1-1.025.275l-2.165-1.25a.75.75 0 1 1 .75-1.3l2.165 1.25A.75.75 0 0 1 4.536 6Z",
        clipRule: "evenodd",
        opacity: 0.3
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M12 14.928a.75.75 0 0 1-1.024-.274l-1.25-2.165a.75.75 0 0 1 1.298-.75l1.25 2.165A.75.75 0 0 1 12 14.928Z",
        clipRule: "evenodd",
        opacity: 0.8
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M6 4.536a.75.75 0 0 1-1.024-.275l-1.25-2.165a.75.75 0 1 1 1.299-.75l1.25 2.165A.75.75 0 0 1 6 4.536Z",
        clipRule: "evenodd",
        opacity: 0.2
    })), Ct = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: 16,
        height: 16,
        fill: "none",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("g", {
        stroke: "currentColor",
        strokeLinecap: "round",
        strokeLinejoin: "round",
        strokeWidth: 1.5
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M13.25 7H9m0 0V2.75M9 7l5.25-5.25M2.75 9H7m0 0v4.25M7 9l-5.25 5.25"
    }))), It = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function({ trackRef: n, ...a }, r) {
    const c = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ab"])(), { mergedProps: s, inFocus: o } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["d"])({
        trackRef: n ?? c,
        props: a
    });
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ac"].Consumer, null, (l)=>l !== void 0 && /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("button", {
            ref: r,
            ...s
        }, a.children ? a.children : o ? /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(Ct, null) : /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(Et, null)));
}), ua = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function({ kind: n, initialSelection: a, onActiveDeviceChange: r, onDeviceListChange: c, onDeviceSelectError: s, exactMatch: o, track: l, requestPermissions: i, onError: u, ...d }, g) {
    const f = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["w"])(), E = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])("default"), p = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "ua.useCallback[p]": (w)=>{
            f && f.emit(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RoomEvent"].MediaDevicesError, w), u == null || u(w);
        }
    }["ua.useCallback[p]"], [
        f,
        u
    ]), { devices: m, activeDeviceId: h, setActiveMediaDevice: y, className: C } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["e"])({
        kind: n,
        room: f,
        track: l,
        requestPermissions: i,
        onError: p
    });
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "ua.useEffect": ()=>{
            a !== void 0 && y(a);
        }
    }["ua.useEffect"], [
        y
    ]), (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "ua.useEffect": ()=>{
            typeof c == "function" && c(m);
        }
    }["ua.useEffect"], [
        c,
        m
    ]), (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "ua.useEffect": ()=>{
            h !== E.current && (r == null || r(h)), E.current = h;
        }
    }["ua.useEffect"], [
        h
    ]);
    const A = async (w)=>{
        try {
            await y(w, {
                exact: o ?? !0
            });
        } catch (M) {
            if (M instanceof Error) s == null || s(M);
            else throw M;
        }
    }, N = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "ua.useMemo[N]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a"])(d, {
                className: C
            }, {
                className: "lk-list"
            })
    }["ua.useMemo[N]"], [
        C,
        d
    ]), v = !!m.find((w)=>w.label.toLowerCase().startsWith("default"));
    function R(w, M, P) {
        return w === M || !v && P === 0 && M === "default";
    }
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("ul", {
        ref: g,
        ...N
    }, m.map((w, M)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("li", {
            key: w.deviceId,
            id: w.deviceId,
            "data-lk-active": R(w.deviceId, h, M),
            "aria-selected": R(w.deviceId, h, M),
            role: "option"
        }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("button", {
            className: "lk-button",
            onClick: ()=>A(w.deviceId)
        }, w.label))));
}), da = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function({ label: n = "Allow Audio", ...a }, r) {
    const c = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["f"])(), { mergedProps: s } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["f"])({
        room: c,
        props: a
    });
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("button", {
        ref: r,
        ...s
    }, n);
}), ma = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function({ label: n, ...a }, r) {
    const c = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["f"])(), { mergedProps: s, canPlayAudio: o } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["f"])({
        room: c,
        props: a
    }), { mergedProps: l, canPlayVideo: i } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["g"])({
        room: c,
        props: s
    }), { style: u, ...d } = l;
    return u.display = o && i ? "none" : "block", /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("button", {
        ref: r,
        style: u,
        ...d
    }, n ?? `Start ${o ? "Video" : "Audio"}`);
});
function ye(t, n) {
    switch(t){
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Microphone:
            return n ? /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(kt, null) : /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(wt, null);
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Camera:
            return n ? /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(vt, null) : /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(gt, null);
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.ScreenShare:
            return n ? /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(St, null) : /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(Me, null);
        default:
            return;
    }
}
function xt(t) {
    switch(t){
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ConnectionQuality"].Excellent:
            return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(Rt, null);
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ConnectionQuality"].Good:
            return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(Mt, null);
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ConnectionQuality"].Poor:
            return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(yt, null);
        default:
            return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(bt, null);
    }
}
const fa = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function({ showIcon: n, ...a }, r) {
    const { buttonProps: c, enabled: s } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["h"])(a), [o, l] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(!1);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "fa.useEffect": ()=>{
            l(!0);
        }
    }["fa.useEffect"], []), o && /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("button", {
        ref: r,
        ...c
    }, (n ?? !0) && ye(a.source, s), a.children);
}), be = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function(n, a) {
    const { className: r, quality: c } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["i"])(n), s = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "be.useMemo[s]": ()=>({
                ...(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a"])(n, {
                    className: r
                }),
                "data-lk-quality": c
            })
    }["be.useMemo[s]"], [
        c,
        n,
        r
    ]);
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("div", {
        ref: a,
        ...s
    }, n.children ?? xt(c));
}), K = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function({ participant: n, ...a }, r) {
    const c = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["c"])(n), { className: s, infoObserver: o } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "K.useMemo": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ad"])(c)
    }["K.useMemo"], [
        c
    ]), { identity: l, name: i } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["j"])(o, {
        name: c.name,
        identity: c.identity,
        metadata: c.metadata
    }), u = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "K.useMemo[u]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a"])(a, {
                className: s,
                "data-lk-participant-name": i
            })
    }["K.useMemo[u]"], [
        a,
        s,
        i
    ]);
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("span", {
        ref: r,
        ...u
    }, i !== "" ? i : l, a.children);
}), Se = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function({ trackRef: n, show: a = "always", ...r }, c) {
    const { className: s, isMuted: o } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["k"])(n), l = a === "always" || a === "muted" && o || a === "unmuted" && !o, i = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Se.useMemo[i]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a"])(r, {
                className: s
            })
    }["Se.useMemo[i]"], [
        s,
        r
    ]);
    return l ? /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("div", {
        ref: c,
        ...i,
        "data-lk-muted": o
    }, r.children ?? ye(n.source, !o)) : null;
}), Pt = (t)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        width: 320,
        height: 320,
        viewBox: "0 0 320 320",
        preserveAspectRatio: "xMidYMid meet",
        fill: "none",
        xmlns: "http://www.w3.org/2000/svg",
        ...t
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M160 180C204.182 180 240 144.183 240 100C240 55.8172 204.182 20 160 20C115.817 20 79.9997 55.8172 79.9997 100C79.9997 144.183 115.817 180 160 180Z",
        fill: "white",
        fillOpacity: 0.25
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("path", {
        d: "M97.6542 194.614C103.267 191.818 109.841 192.481 115.519 195.141C129.025 201.466 144.1 205 159.999 205C175.899 205 190.973 201.466 204.48 195.141C210.158 192.481 216.732 191.818 222.345 194.614C262.703 214.719 291.985 253.736 298.591 300.062C300.15 310.997 291.045 320 280 320H39.9997C28.954 320 19.8495 310.997 21.4087 300.062C28.014 253.736 57.2966 214.72 97.6542 194.614Z",
        fill: "white",
        fillOpacity: 0.25
    }));
function Ce(t, n = {}) {
    const [a, r] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])((0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["X"])(t)), [c, s] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(a == null ? void 0 : a.isMuted), [o, l] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(a == null ? void 0 : a.isSubscribed), [i, u] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(a == null ? void 0 : a.track), [d, g] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])("landscape"), f = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(), { className: E, trackObserver: p } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Ce.useMemo": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Y"])(t)
    }["Ce.useMemo"], [
        t.participant.sid ?? t.participant.identity,
        t.source,
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a1"])(t) && t.publication.trackSid
    ]);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Ce.useEffect": ()=>{
            const m = p.subscribe({
                "Ce.useEffect.m": (h)=>{
                    __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].debug("update track", h), r(h), s(h == null ? void 0 : h.isMuted), l(h == null ? void 0 : h.isSubscribed), u(h == null ? void 0 : h.track);
                }
            }["Ce.useEffect.m"]);
            return ({
                "Ce.useEffect": ()=>m == null ? void 0 : m.unsubscribe()
            })["Ce.useEffect"];
        }
    }["Ce.useEffect"], [
        p
    ]), (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Ce.useEffect": ()=>{
            var m, h;
            return i && (f.current && i.detach(f.current), (m = n.element) != null && m.current && !(t.participant.isLocal && (i == null ? void 0 : i.kind) === "audio") && i.attach(n.element.current)), f.current = (h = n.element) == null ? void 0 : h.current, ({
                "Ce.useEffect": ()=>{
                    f.current && (i == null || i.detach(f.current));
                }
            })["Ce.useEffect"];
        }
    }["Ce.useEffect"], [
        i,
        n.element
    ]), (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Ce.useEffect": ()=>{
            var m, h;
            if (typeof ((m = a == null ? void 0 : a.dimensions) == null ? void 0 : m.width) == "number" && typeof ((h = a == null ? void 0 : a.dimensions) == null ? void 0 : h.height) == "number") {
                const y = a.dimensions.width > a.dimensions.height ? "landscape" : "portrait";
                g(y);
            }
        }
    }["Ce.useEffect"], [
        a
    ]), {
        publication: a,
        isMuted: c,
        isSubscribed: o,
        track: i,
        elementProps: (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a"])(n.props, {
            className: E,
            "data-lk-local-participant": t.participant.isLocal,
            "data-lk-source": a == null ? void 0 : a.source,
            ...(a == null ? void 0 : a.kind) === "video" && {
                "data-lk-orientation": d
            }
        })
    };
}
var Y, ue;
function Tt() {
    if (ue) return Y;
    ue = 1;
    var t = "Expected a function", n = NaN, a = "[object Symbol]", r = /^\s+|\s+$/g, c = /^[-+]0x[0-9a-f]+$/i, s = /^0b[01]+$/i, o = /^0o[0-7]+$/i, l = parseInt, i = typeof __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ae"] == "object" && __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ae"] && __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ae"].Object === Object && __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ae"], u = typeof self == "object" && self && self.Object === Object && self, d = i || u || Function("return this")(), g = Object.prototype, f = g.toString, E = Math.max, p = Math.min, m = function() {
        return d.Date.now();
    };
    function h(v, R, w) {
        var M, P, O, L, b, T, F = 0, re = !1, Z = !1, D = !0;
        if (typeof v != "function") throw new TypeError(t);
        R = N(R) || 0, y(w) && (re = !!w.leading, Z = "maxWait" in w, O = Z ? E(N(w.maxWait) || 0, R) : O, D = "trailing" in w ? !!w.trailing : D);
        function U(k) {
            var x = M, H = P;
            return M = P = void 0, F = k, L = v.apply(H, x), L;
        }
        function Pe(k) {
            return F = k, b = setTimeout(W, R), re ? U(k) : L;
        }
        function Te(k) {
            var x = k - T, H = k - F, se = R - x;
            return Z ? p(se, O - H) : se;
        }
        function ce(k) {
            var x = k - T, H = k - F;
            return T === void 0 || x >= R || x < 0 || Z && H >= O;
        }
        function W() {
            var k = m();
            if (ce(k)) return le(k);
            b = setTimeout(W, Te(k));
        }
        function le(k) {
            return b = void 0, D && M ? U(k) : (M = P = void 0, L);
        }
        function Ae() {
            b !== void 0 && clearTimeout(b), F = 0, M = T = P = b = void 0;
        }
        function Ne() {
            return b === void 0 ? L : le(m());
        }
        function G() {
            var k = m(), x = ce(k);
            if (M = arguments, P = this, T = k, x) {
                if (b === void 0) return Pe(T);
                if (Z) return b = setTimeout(W, R), U(T);
            }
            return b === void 0 && (b = setTimeout(W, R)), L;
        }
        return G.cancel = Ae, G.flush = Ne, G;
    }
    function y(v) {
        var R = typeof v;
        return !!v && (R == "object" || R == "function");
    }
    function C(v) {
        return !!v && typeof v == "object";
    }
    function A(v) {
        return typeof v == "symbol" || C(v) && f.call(v) == a;
    }
    function N(v) {
        if (typeof v == "number") return v;
        if (A(v)) return n;
        if (y(v)) {
            var R = typeof v.valueOf == "function" ? v.valueOf() : v;
            v = y(R) ? R + "" : R;
        }
        if (typeof v != "string") return v === 0 ? v : +v;
        v = v.replace(r, "");
        var w = s.test(v);
        return w || o.test(v) ? l(v.slice(2), w ? 2 : 8) : c.test(v) ? n : +v;
    }
    return Y = h, Y;
}
var At = Tt();
const de = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["af"])(At);
function Nt(t) {
    const n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(t);
    n.current = t, (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])(()=>()=>{
            n.current();
        }, []);
}
function Lt(t, n = 500, a) {
    const r = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])();
    Nt(()=>{
        r.current && r.current.cancel();
    });
    const c = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])(()=>{
        const s = de(t, n, a), o = (...l)=>s(...l);
        return o.cancel = ()=>{
            s.cancel();
        }, o.isPending = ()=>!!r.current, o.flush = ()=>s.flush(), o;
    }, [
        t,
        n,
        a
    ]);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        r.current = de(t, n, a);
    }, [
        t,
        n,
        a
    ]), c;
}
function zt(t, n, a) {
    const r = (u, d)=>u === d, c = t instanceof Function ? t() : t, [s, o] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(c), l = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(c), i = Lt(o, n, a);
    return r(l.current, c) || (i(c), l.current = c), [
        s,
        i
    ];
}
function Vt({ threshold: t = 0, root: n = null, rootMargin: a = "0%", freezeOnceVisible: r = !1, initialIsIntersecting: c = !1, onChange: s } = {}) {
    var o;
    const [l, i] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(null), [u, d] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(()=>({
            isIntersecting: c,
            entry: void 0
        })), g = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])();
    g.current = s;
    const f = ((o = u.entry) == null ? void 0 : o.isIntersecting) && r;
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        if (!l || !("IntersectionObserver" in window) || f) return;
        const m = new IntersectionObserver((h)=>{
            const y = Array.isArray(m.thresholds) ? m.thresholds : [
                m.thresholds
            ];
            h.forEach((C)=>{
                const A = C.isIntersecting && y.some((N)=>C.intersectionRatio >= N);
                d({
                    isIntersecting: A,
                    entry: C
                }), g.current && g.current(A, C);
            });
        }, {
            threshold: t,
            root: n,
            rootMargin: a
        });
        return m.observe(l), ()=>{
            m.disconnect();
        };
    }, [
        l,
        // eslint-disable-next-line react-hooks/exhaustive-deps
        JSON.stringify(t),
        n,
        a,
        f,
        r
    ]);
    const E = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        var m;
        !l && (m = u.entry) != null && m.target && !r && !f && E.current !== u.entry.target && (E.current = u.entry.target, d({
            isIntersecting: c,
            entry: void 0
        }));
    }, [
        l,
        u.entry,
        r,
        f,
        c
    ]);
    const p = [
        i,
        !!u.isIntersecting,
        u.entry
    ];
    return p.ref = p[0], p.isIntersecting = p[1], p.entry = p[2], p;
}
const Ft = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function({ onTrackClick: n, onClick: a, onSubscriptionStatusChanged: r, trackRef: c, manageSubscription: s, ...o }, l) {
    const i = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["i"])(c), u = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useImperativeHandle"])(l, {
        "Ft.useImperativeHandle": ()=>u.current
    }["Ft.useImperativeHandle"]);
    const d = Vt({
        root: u.current
    }), [g] = zt(d, 3e3);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Ft.useEffect": ()=>{
            s && i.publication instanceof __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RemoteTrackPublication"] && (g == null ? void 0 : g.isIntersecting) === !1 && (d == null ? void 0 : d.isIntersecting) === !1 && i.publication.setSubscribed(!1);
        }
    }["Ft.useEffect"], [
        g,
        i,
        s
    ]), (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Ft.useEffect": ()=>{
            s && i.publication instanceof __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RemoteTrackPublication"] && (d == null ? void 0 : d.isIntersecting) === !0 && i.publication.setSubscribed(!0);
        }
    }["Ft.useEffect"], [
        d,
        i,
        s
    ]);
    const { elementProps: f, publication: E, isSubscribed: p } = Ce(i, {
        element: u,
        props: o
    });
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Ft.useEffect": ()=>{
            r == null || r(!!p);
        }
    }["Ft.useEffect"], [
        p,
        r
    ]);
    const m = (h)=>{
        a == null || a(h), n == null || n({
            participant: i == null ? void 0 : i.participant,
            track: E
        });
    };
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("video", {
        ref: u,
        ...f,
        muted: !0,
        onClick: m
    });
}), ne = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function({ trackRef: n, onSubscriptionStatusChanged: a, volume: r, ...c }, s) {
    const o = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["i"])(n), l = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useImperativeHandle"])(s, {
        "ne.useImperativeHandle": ()=>l.current
    }["ne.useImperativeHandle"]);
    const { elementProps: i, isSubscribed: u, track: d, publication: g } = Ce(o, {
        element: l,
        props: c
    });
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "ne.useEffect": ()=>{
            a == null || a(!!u);
        }
    }["ne.useEffect"], [
        u,
        a
    ]), (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "ne.useEffect": ()=>{
            d === void 0 || r === void 0 || (d instanceof __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RemoteAudioTrack"] ? d.setVolume(r) : __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].warn("Volume can only be set on remote audio tracks."));
        }
    }["ne.useEffect"], [
        r,
        d
    ]), (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "ne.useEffect": ()=>{
            g === void 0 || c.muted === void 0 || (g instanceof __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["RemoteTrackPublication"] ? g.setEnabled(!c.muted) : __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].warn("Can only call setEnabled on remote track publications."));
        }
    }["ne.useEffect"], [
        c.muted,
        g,
        d
    ]), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("audio", {
        ref: l,
        ...i
    });
});
function Zt(t) {
    const n = !!(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["B"])();
    return t.participant && !n ? /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ag"].Provider, {
        value: t.participant
    }, t.children) : /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], null, t.children);
}
function Ht(t) {
    const n = !!(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ab"])();
    return t.trackRef && !n ? /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ah"].Provider, {
        value: t.trackRef
    }, t.children) : /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], null, t.children);
}
const Bt = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function({ trackRef: n, children: a, onParticipantClick: r, disableSpeakingIndicator: c, ...s }, o) {
    var E, p;
    const l = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["i"])(n), { elementProps: i } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"])({
        htmlProps: s,
        disableSpeakingIndicator: c,
        onParticipantClick: r,
        trackRef: l
    }), u = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["m"])(l.participant), d = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["j"])(), g = (E = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ai"])()) == null ? void 0 : E.autoSubscription, f = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "Bt.useCallback[f]": (m)=>{
            l.source && !m && d && d.pin.dispatch && (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["m"])(l, d.pin.state) && d.pin.dispatch({
                msg: "clear_pin"
            });
        }
    }["Bt.useCallback[f]"], [
        l,
        d
    ]);
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("div", {
        ref: o,
        style: {
            position: "relative"
        },
        ...i
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(Ht, {
        trackRef: l
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(Zt, {
        participant: l.participant
    }, a ?? /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], null, (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a1"])(l) && (((p = l.publication) == null ? void 0 : p.kind) === "video" || l.source === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Camera || l.source === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.ScreenShare) ? /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(Ft, {
        trackRef: l,
        onSubscriptionStatusChanged: f,
        manageSubscription: g
    }) : (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a1"])(l) && /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(ne, {
        trackRef: l,
        onSubscriptionStatusChanged: f
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("div", {
        className: "lk-participant-placeholder"
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(Pt, null)), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("div", {
        className: "lk-participant-metadata"
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("div", {
        className: "lk-participant-metadata-item"
    }, l.source === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Camera ? /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], null, u && /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(pt, {
        style: {
            marginRight: "0.25rem"
        }
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(Se, {
        trackRef: {
            participant: l.participant,
            source: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Microphone
        },
        show: "muted"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(K, null)) : /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], null, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(Me, {
        style: {
            marginRight: "0.25rem"
        }
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(K, null, "'s screen"))), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(be, {
        className: "lk-participant-metadata-item"
    }))), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(It, {
        trackRef: l
    }))));
});
function ha(t) {
    const n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a"])(t, {
        className: "lk-focus-layout"
    });
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("div", {
        ...n
    }, t.children);
}
function ga({ trackRef: t, ...n }) {
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(Bt, {
        trackRef: t,
        ...n
    });
}
function Ie({ tracks: t, ...n }) {
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], null, t.map((a)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ah"].Provider, {
            value: a,
            key: (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["p"])(a)
        }, (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["c"])(n.children))));
}
function jt({ totalPageCount: t, nextPage: n, prevPage: a, currentPage: r, pagesContainer: c }) {
    const [s, o] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(!1);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "jt.useEffect": ()=>{
            let l;
            return c && (l = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["aj"])(c.current, 2e3).subscribe(o)), ({
                "jt.useEffect": ()=>{
                    l && l.unsubscribe();
                }
            })["jt.useEffect"];
        }
    }["jt.useEffect"], [
        c
    ]), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("div", {
        className: "lk-pagination-control",
        "data-lk-user-interaction": s
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("button", {
        className: "lk-button",
        onClick: a
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(ie, null)), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("span", {
        className: "lk-pagination-count"
    }, `${r} of ${t}`), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("button", {
        className: "lk-button",
        onClick: n
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(ie, null)));
}
const _t = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function({ totalPageCount: n, currentPage: a }, r) {
    const c = new Array(n).fill("").map((s, o)=>o + 1 === a ? /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("span", {
            "data-lk-active": !0,
            key: o
        }) : /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("span", {
            key: o
        }));
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("div", {
        ref: r,
        className: "lk-pagination-indicator"
    }, c);
});
function va({ tracks: t, ...n }) {
    const a = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createRef"])(), r = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "va.useMemo[r]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a"])(n, {
                className: "lk-grid-layout"
            })
    }["va.useMemo[r]"], [
        n
    ]), { layout: c } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["n"])(a, t.length), s = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["o"])(c.maxTiles, t);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["p"])(a, {
        onLeftSwipe: s.nextPage,
        onRightSwipe: s.prevPage
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("div", {
        ref: a,
        "data-lk-pagination": s.totalPageCount > 1,
        ...r
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(Ie, {
        tracks: s.tracks
    }, n.children), t.length > c.maxTiles && /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], null, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(_t, {
        totalPageCount: s.totalPageCount,
        currentPage: s.currentPage
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(jt, {
        pagesContainer: a,
        ...s
    })));
}
const Ot = 130, Wt = 140, me = 1, xe = 16 / 10, qt = (1 - xe) * -1;
function Ea({ tracks: t, orientation: n, ...a }) {
    const r = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null), [c, s] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(0), { width: o, height: l } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["q"])(r), i = n || (l >= o ? "vertical" : "horizontal"), u = i === "vertical" ? Math.max(o * qt, Ot) : Math.max(l * xe, Wt), d = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ak"])(), g = Math.max(i === "vertical" ? (l - d) / u : (o - d) / u, me);
    let f = Math.round(g);
    Math.abs(g - c) < 0.5 ? f = Math.round(c) : c !== g && s(g);
    const E = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["r"])(t, f);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useLayoutEffect"])({
        "Ea.useLayoutEffect": ()=>{
            r.current && (r.current.dataset.lkOrientation = i, r.current.style.setProperty("--lk-max-visible-tiles", f.toString()));
        }
    }["Ea.useLayoutEffect"], [
        f,
        i
    ]), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("aside", {
        key: i,
        className: "lk-carousel",
        ref: r,
        ...a
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(Ie, {
        tracks: E
    }, a.children));
}
function pa({ value: t, onPinChange: n, onWidgetChange: a, children: r }) {
    const c = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["al"])(t);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "pa.useEffect": ()=>{
            __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].debug("PinState Updated", {
                state: c.pin.state
            }), n && c.pin.state && n(c.pin.state);
        }
    }["pa.useEffect"], [
        c.pin.state,
        n
    ]), (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "pa.useEffect": ()=>{
            __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"].debug("Widget Updated", {
                widgetState: c.widget.state
            }), a && c.widget.state && a(c.widget.state);
        }
    }["pa.useEffect"], [
        a,
        c.widget.state
    ]), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ac"].Provider, {
        value: c
    }, r);
}
const wa = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function({ trackRef: n, ...a }, r) {
    const d = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["i"])(n), g = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["s"])(d, {
        bands: 7,
        loPass: 300
    });
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("svg", {
        ref: r,
        width: "100%",
        height: "100%",
        viewBox: "0 0 200 90",
        ...a,
        className: "lk-audio-visualizer"
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("rect", {
        x: "0",
        y: "0",
        width: "100%",
        height: "100%"
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("g", {
        style: {
            transform: `translate(${(200 - 7 * 10) / 2}px, 0)`
        }
    }, g.map((f, E)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("rect", {
            key: E,
            x: E * 10,
            y: 90 / 2 - f * 50 / 2,
            width: 6,
            height: f * 50
        }))));
});
function ka({ participants: t, ...n }) {
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], null, t.map((a)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ag"].Provider, {
            value: a,
            key: a.identity
        }, (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["c"])(n.children))));
}
function Ra({ volume: t, muted: n }) {
    const a = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["t"])([
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Microphone,
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.ScreenShareAudio,
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Source.Unknown
    ], {
        updateOnlyOn: [],
        onlySubscribed: !0
    }).filter((r)=>!r.participant.isLocal && r.publication.kind === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Track"].Kind.Audio);
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("div", {
        style: {
            display: "none"
        }
    }, a.map((r)=>/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(ne, {
            key: (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["p"])(r),
            trackRef: r,
            volume: t,
            muted: n
        })));
}
const Ma = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function({ childrenPosition: n = "before", children: a, ...r }, c) {
    const { name: s } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["v"])();
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("span", {
        ref: c,
        ...r
    }, n === "before" && a, s, n === "after" && a);
});
function $t(t) {
    const n = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "$t.useMemo[n]": ()=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a"])(t, {
                className: "lk-toast"
            })
    }["$t.useMemo[n]"], [
        t
    ]);
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("div", {
        ...n
    }, t.children);
}
const Dt = (t)=>{
    const n = [];
    for(let a = 0; a < t; a++)n.push([
        a,
        t - 1 - a
    ]);
    return n;
}, fe = (t)=>[
        [
            Math.floor(t / 2)
        ],
        [
            -1
        ]
    ], Ut = (t, n, a)=>{
    const [r, c] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(0), [s, o] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])([
        []
    ]);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        if (t === "thinking") o(fe(n));
        else if (t === "connecting" || t === "initializing") {
            const i = [
                ...Dt(n)
            ];
            o(i);
        } else o(t === "listening" ? fe(n) : t === void 0 || t === "speaking" ? [
            new Array(n).fill(0).map((i, u)=>u)
        ] : [
            []
        ]);
        c(0);
    }, [
        t,
        n
    ]);
    const l = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        let i = performance.now();
        const u = (d)=>{
            d - i >= a && (c((f)=>f + 1), i = d), l.current = requestAnimationFrame(u);
        };
        return l.current = requestAnimationFrame(u), ()=>{
            l.current !== null && cancelAnimationFrame(l.current);
        };
    }, [
        a,
        n,
        t,
        s.length
    ]), s[r % s.length];
}, Gt = /* @__PURE__ */ new Map([
    [
        "connecting",
        2e3
    ],
    [
        "initializing",
        2e3
    ],
    [
        "listening",
        500
    ],
    [
        "thinking",
        150
    ]
]), Qt = (t, n)=>{
    if (t === void 0) return 1e3;
    let a = Gt.get(t);
    if (a) switch(t){
        case "connecting":
            a /= n;
            break;
    }
    return a;
}, Xt = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function({ state: n, options: a, barCount: r = 15, trackRef: c, children: s, ...o }, l) {
    const i = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a"])(o, {
        className: "lk-audio-bar-visualizer"
    });
    let u = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ab"])();
    c && (u = c);
    const d = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["s"])(u, {
        bands: r,
        loPass: 100,
        hiPass: 200
    }), g = (a == null ? void 0 : a.minHeight) ?? 20, f = (a == null ? void 0 : a.maxHeight) ?? 100, E = Ut(n, r, Qt(n, r) ?? 100);
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("div", {
        ref: l,
        ...i,
        "data-lk-va-state": n
    }, d.map((p, m)=>s ? (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$room$2d$DhBnHppi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["c"])(s, {
            "data-lk-highlighted": E.includes(m),
            "data-lk-bar-index": m,
            className: "lk-audio-bar",
            style: {
                height: `${Math.min(f, Math.max(g, p * 100 + 5))}%`
            }
        }) : /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("span", {
            key: m,
            "data-lk-highlighted": E.includes(m),
            "data-lk-bar-index": m,
            className: `lk-audio-bar ${E.includes(m) && "lk-highlighted"}`,
            style: {
                // TODO transform animations would be more performant, however the border-radius gets distorted when using scale transforms. a 9-slice approach (or 3 in this case) could work
                // transform: `scale(1, ${Math.min(maxHeight, Math.max(minHeight, volume))}`,
                height: `${Math.min(f, Math.max(g, p * 100 + 5))}%`
            }
        })));
}), ya = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function({ children: n, disableSpeakingIndicator: a, onParticipantClick: r, trackRef: c, ...s }, o) {
    const l = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["i"])(c), { elementProps: i } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["l"])({
        trackRef: l,
        htmlProps: s,
        disableSpeakingIndicator: a,
        onParticipantClick: r
    });
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("div", {
        ref: o,
        style: {
            position: "relative",
            minHeight: "160px"
        },
        ...i
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ah"].Provider, {
        value: l
    }, n ?? /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], null, (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a1"])(l) && /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(ne, {
        trackRef: l
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(Xt, {
        barCount: 7,
        options: {
            minHeight: 8
        }
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("div", {
        className: "lk-participant-metadata"
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("div", {
        className: "lk-participant-metadata-item"
    }, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(Se, {
        trackRef: l
    }), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(K, null)), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(be, {
        className: "lk-participant-metadata-item"
    })))));
});
function ba(t) {
    const [n, a] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(void 0), r = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["a"])(t.room);
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "ba.useEffect": ()=>{
            switch(r){
                case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ConnectionState"].Reconnecting:
                    a(/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], null, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(oe, {
                        className: "lk-spinner"
                    }), " Reconnecting"));
                    break;
                case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ConnectionState"].Connecting:
                    a(/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], null, /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(oe, {
                        className: "lk-spinner"
                    }), " Connecting"));
                    break;
                case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ConnectionState"].Disconnected:
                    a(/* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], null, "Disconnected"));
                    break;
                default:
                    a(void 0);
                    break;
            }
        }
    }["ba.useEffect"], [
        r
    ]), n ? /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])($t, {
        className: "lk-toast-connection-state"
    }, n) : /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], null);
}
const Sa = /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["forwardRef"])(function({ entry: n, hideName: a = !1, hideTimestamp: r = !1, messageFormatter: c, ...s }, o) {
    var f, E, p, m;
    const l = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "Sa.useMemo[l]": ()=>c ? c(n.message) : n.message
    }["Sa.useMemo[l]"], [
        n.message,
        c
    ]), i = !!n.editTimestamp, u = new Date(n.timestamp), d = typeof navigator < "u" ? navigator.language : "en-US", g = ((f = n.from) == null ? void 0 : f.name) ?? ((E = n.from) == null ? void 0 : E.identity);
    return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("li", {
        ref: o,
        className: "lk-chat-entry",
        title: u.toLocaleTimeString(d, {
            timeStyle: "full"
        }),
        "data-lk-message-origin": (p = n.from) != null && p.isLocal ? "local" : "remote",
        ...s
    }, (!r || !a || i) && /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("span", {
        className: "lk-meta-data"
    }, !a && /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("strong", {
        className: "lk-participant-name"
    }, g), (!r || i) && /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("span", {
        className: "lk-timestamp"
    }, i && "edited ", u.toLocaleTimeString(d, {
        timeStyle: "short"
    }))), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("span", {
        className: "lk-message-body"
    }, l), /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("span", {
        className: "lk-message-attachements"
    }, (m = n.attachedFiles) == null ? void 0 : m.map((h)=>h.type.startsWith("image/") && /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("img", {
            style: {
                maxWidth: "300px",
                maxHeight: "300px"
            },
            key: h.name,
            src: URL.createObjectURL(h),
            alt: h.name
        }))));
});
function Ca(t) {
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["am"])(t, (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["an"])()).map((n, a)=>{
        if (typeof n == "string") return n;
        {
            const r = n.content.toString(), c = n.type === "url" ? /^http(s?):\/\//.test(r) ? r : `https://${r}` : `mailto:${r}`;
            return /* @__PURE__ */ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createElement"])("a", {
                className: "lk-chat-link",
                key: a,
                href: c,
                target: "_blank",
                rel: "noreferrer"
            }, r);
        }
    });
}
;
 //# sourceMappingURL=components-k0KtCs0w.mjs.map
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/components-k0KtCs0w.mjs [app-client] (ecmascript) <export R as RoomAudioRenderer>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "RoomAudioRenderer": (()=>__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["R"])
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/components-k0KtCs0w.mjs [app-client] (ecmascript)");
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/contexts-CH4CA2DN.mjs [app-client] (ecmascript) <export R as RoomContext>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "RoomContext": (()=>__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["R"])
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/contexts-CH4CA2DN.mjs [app-client] (ecmascript)");
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/components-k0KtCs0w.mjs [app-client] (ecmascript) <export q as StartAudio>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "StartAudio": (()=>__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["q"])
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/components-k0KtCs0w.mjs [app-client] (ecmascript)");
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/contexts-CH4CA2DN.mjs [app-client] (ecmascript) <export f as useRoomContext>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "useRoomContext": (()=>__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["f"])
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/contexts-CH4CA2DN.mjs [app-client] (ecmascript)");
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript) <export V as useVoiceAssistant>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "useVoiceAssistant": (()=>__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["V"])
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript)");
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/components-k0KtCs0w.mjs [app-client] (ecmascript) <export B as BarVisualizer>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "BarVisualizer": (()=>__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["B"])
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/components-k0KtCs0w.mjs [app-client] (ecmascript)");
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript) <export M as useRemoteParticipants>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "useRemoteParticipants": (()=>__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["M"])
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript)");
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/contexts-CH4CA2DN.mjs [app-client] (ecmascript) <export w as useMaybeRoomContext>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "useMaybeRoomContext": (()=>__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["w"])
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/contexts-CH4CA2DN.mjs [app-client] (ecmascript)");
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript) <export e as useMediaDeviceSelect>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "useMediaDeviceSelect": (()=>__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["e"])
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript)");
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript) <export C as useLocalParticipant>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "useLocalParticipant": (()=>__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["C"])
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript)");
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript) <export x as usePersistentUserChoices>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "usePersistentUserChoices": (()=>__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["x"])
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript)");
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript) <export h as useTrackToggle>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "useTrackToggle": (()=>__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["h"])
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript)");
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript) <export A as useLocalParticipantPermissions>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "useLocalParticipantPermissions": (()=>__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["A"])
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript)");
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript) <export t as useTracks>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "useTracks": (()=>__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["t"])
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript)");
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/components-k0KtCs0w.mjs [app-client] (ecmascript) <export V as VideoTrack>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "VideoTrack": (()=>__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["V"])
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/components-k0KtCs0w.mjs [app-client] (ecmascript)");
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript) <export w as useChat>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "useChat": (()=>__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["w"])
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript)");
}}),
"[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript) <export _ as useTranscriptions>": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "useTranscriptions": (()=>__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__["_"])
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-client] (ecmascript)");
}}),
}]);

//# sourceMappingURL=cd8b4_%40livekit_components-react_dist_36e1cbeb._.js.map