module.exports = {

"[project]/components/ui/alert.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "Alert": (()=>Alert),
    "AlertDescription": (()=>AlertDescription),
    "AlertTitle": (()=>AlertTitle)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$class$2d$variance$2d$authority$40$0$2e$7$2e$1$2f$node_modules$2f$class$2d$variance$2d$authority$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/class-variance-authority@0.7.1/node_modules/class-variance-authority/dist/index.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/utils.ts [app-ssr] (ecmascript)");
;
;
;
const alertVariants = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$class$2d$variance$2d$authority$40$0$2e$7$2e$1$2f$node_modules$2f$class$2d$variance$2d$authority$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cva"])([
    'relative w-full rounded-lg border px-4 py-3 text-sm grid grid-cols-[0_1fr] gap-y-0.5 items-start',
    'has-[>svg]:grid-cols-[calc(var(--spacing)*4)_1fr] has-[>svg]:gap-x-3 [&>svg]:size-4 [&>svg]:translate-y-0.5 [&>svg]:text-current'
], {
    variants: {
        variant: {
            default: 'bg-card text-card-foreground',
            destructive: [
                'text-destructive-foreground bg-destructive border-destructive-border',
                '[&>svg]:text-current *:data-[slot=alert-description]:text-destructive-foreground/90'
            ]
        }
    },
    defaultVariants: {
        variant: 'default'
    }
});
function Alert({ className, variant, ...props }) {
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        "data-slot": "alert",
        role: "alert",
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])(alertVariants({
            variant
        }), className),
        ...props
    }, void 0, false, {
        fileName: "[project]/components/ui/alert.tsx",
        lineNumber: 32,
        columnNumber: 5
    }, this);
}
function AlertTitle({ className, ...props }) {
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        "data-slot": "alert-title",
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])('col-start-2 line-clamp-1 min-h-4 font-medium tracking-tight', className),
        ...props
    }, void 0, false, {
        fileName: "[project]/components/ui/alert.tsx",
        lineNumber: 43,
        columnNumber: 5
    }, this);
}
function AlertDescription({ className, ...props }) {
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        "data-slot": "alert-description",
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])('text-muted-foreground col-start-2 grid justify-items-start gap-1 text-sm [&_p]:leading-relaxed', className),
        ...props
    }, void 0, false, {
        fileName: "[project]/components/ui/alert.tsx",
        lineNumber: 53,
        columnNumber: 5
    }, this);
}
;
}}),
"[project]/components/alert-toast.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "toastAlert": (()=>toastAlert)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$sonner$40$2$2e$0$2e$6_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$sonner$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/sonner@2.0.6_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/sonner/dist/index.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$Warning$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@phosphor-icons+react@2.1.10_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/@phosphor-icons/react/dist/ssr/Warning.es.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$alert$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/ui/alert.tsx [app-ssr] (ecmascript)");
'use client';
;
;
;
;
function toastAlert(toast) {
    return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$sonner$40$2$2e$0$2e$6_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$sonner$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["toast"].custom((id)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(AlertToast, {
            id: id,
            title: toast.title,
            description: toast.description
        }, void 0, false, {
            fileName: "[project]/components/alert-toast.tsx",
            lineNumber: 16,
            columnNumber: 13
        }, this), {
        duration: 10_000
    });
}
function AlertToast(props) {
    const { title, description, id } = props;
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$alert$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Alert"], {
        onClick: ()=>__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$sonner$40$2$2e$0$2e$6_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$sonner$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["toast"].dismiss(id),
        className: "bg-accent",
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$Warning$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["WarningIcon"], {
                weight: "bold"
            }, void 0, false, {
                fileName: "[project]/components/alert-toast.tsx",
                lineNumber: 26,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$alert$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["AlertTitle"], {
                children: title
            }, void 0, false, {
                fileName: "[project]/components/alert-toast.tsx",
                lineNumber: 27,
                columnNumber: 7
            }, this),
            description && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$alert$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["AlertDescription"], {
                children: description
            }, void 0, false, {
                fileName: "[project]/components/alert-toast.tsx",
                lineNumber: 28,
                columnNumber: 23
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/components/alert-toast.tsx",
        lineNumber: 25,
        columnNumber: 5
    }, this);
}
}}),
"[project]/components/ui/button.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "Button": (()=>Button),
    "buttonVariants": (()=>buttonVariants)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$class$2d$variance$2d$authority$40$0$2e$7$2e$1$2f$node_modules$2f$class$2d$variance$2d$authority$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/class-variance-authority@0.7.1/node_modules/class-variance-authority/dist/index.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$slot$40$1$2e$2$2e$3_$40$types$2b$react$40$19$2e$1$2e$8_react$40$19$2e$1$2e$0$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$slot$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@radix-ui+react-slot@1.2.3_@types+react@19.1.8_react@19.1.0/node_modules/@radix-ui/react-slot/dist/index.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/utils.ts [app-ssr] (ecmascript)");
;
;
;
;
const buttonVariants = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$class$2d$variance$2d$authority$40$0$2e$7$2e$1$2f$node_modules$2f$class$2d$variance$2d$authority$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cva"])([
    'text-xs font-bold tracking-wider uppercase whitespace-nowrap',
    'inline-flex items-center justify-center gap-2 shrink-0 rounded-full cursor-pointer outline-none transition-colors duration-300',
    'focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]',
    'disabled:pointer-events-none disabled:opacity-50',
    'aria-invalid:ring-destructive/20 aria-invalid:border-destructive dark:aria-invalid:ring-destructive/40 ',
    "[&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 [&_svg]:shrink-0"
], {
    variants: {
        variant: {
            default: 'bg-button text-button-foreground hover:bg-muted focus:bg-muted',
            destructive: [
                'bg-destructive text-destructive-foreground',
                'hover:bg-destructive-hover focus:bg-destructive-hover focus-visible:ring-destructive-foreground/20',
                'dark:focus-visible:ring-destructive-foreground/40'
            ],
            outline: [
                'border bg-background',
                'hover:bg-accent hover:text-accent-foreground',
                'dark:bg-input/30 dark:border-input dark:hover:bg-input/50'
            ],
            primary: 'bg-primary text-primary-foreground hover:bg-primary-hover focus:bg-primary-hover',
            secondary: 'bg-secondary text-secondary-foregroun hover:bg-secondary/80',
            ghost: 'hover:bg-accent hover:text-accent-foreground dark:hover:bg-accent/50',
            link: 'text-primary underline-offset-4 hover:underline'
        },
        size: {
            default: 'h-9 px-4 py-2 has-[>svg]:px-3',
            sm: 'h-8 gap-1.5 px-3 has-[>svg]:px-2.5',
            lg: 'h-10 px-6 has-[>svg]:px-4',
            icon: 'size-9'
        }
    },
    defaultVariants: {
        variant: 'default',
        size: 'default'
    }
});
function Button({ className, variant, size, asChild = false, ...props }) {
    const Comp = asChild ? __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$slot$40$1$2e$2$2e$3_$40$types$2b$react$40$19$2e$1$2e$8_react$40$19$2e$1$2e$0$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$slot$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Slot"] : 'button';
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(Comp, {
        "data-slot": "button",
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])(buttonVariants({
            variant,
            size,
            className
        })),
        ...props
    }, void 0, false, {
        fileName: "[project]/components/ui/button.tsx",
        lineNumber: 61,
        columnNumber: 5
    }, this);
}
;
}}),
"[project]/components/livekit/chat/chat-input.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "ChatInput": (()=>ChatInput)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$button$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/ui/button.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/utils.ts [app-ssr] (ecmascript)");
;
;
;
;
function ChatInput({ onSend, className, disabled, ...props }) {
    const inputRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useRef"])(null);
    const [message, setMessage] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])('');
    const handleSubmit = (e)=>{
        e.preventDefault();
        props.onSubmit?.(e);
        onSend?.(message);
        setMessage('');
    };
    const isDisabled = disabled || message.trim().length === 0;
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        if (disabled) return;
        // when not disabled refocus on input
        inputRef.current?.focus();
    }, [
        disabled
    ]);
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("form", {
        ...props,
        onSubmit: handleSubmit,
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])('flex items-center gap-2 rounded-md pl-1 text-sm', className),
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                autoFocus: true,
                ref: inputRef,
                type: "text",
                value: message,
                disabled: disabled,
                placeholder: "Type something...",
                onChange: (e)=>setMessage(e.target.value),
                className: "flex-1 focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
            }, void 0, false, {
                fileName: "[project]/components/livekit/chat/chat-input.tsx",
                lineNumber: 35,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$button$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Button"], {
                size: "sm",
                type: "submit",
                variant: isDisabled ? 'secondary' : 'primary',
                disabled: isDisabled,
                className: "font-mono",
                children: "SEND"
            }, void 0, false, {
                fileName: "[project]/components/livekit/chat/chat-input.tsx",
                lineNumber: 45,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/components/livekit/chat/chat-input.tsx",
        lineNumber: 30,
        columnNumber: 5
    }, this);
}
}}),
"[project]/components/ui/toggle.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "Toggle": (()=>Toggle),
    "toggleVariants": (()=>toggleVariants)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$class$2d$variance$2d$authority$40$0$2e$7$2e$1$2f$node_modules$2f$class$2d$variance$2d$authority$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/class-variance-authority@0.7.1/node_modules/class-variance-authority/dist/index.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$toggle$40$1$2e$1$2e$9_$40$types$2b$react$2d$dom$40$19$2e$1$2e$6_$40$types$2b$react$40$19$2e$1$2e$8_$5f40$types$2b$react$40$19$2e$1$2e$8_qow4awvdp5vvbaipk7nw4amefq$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$toggle$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@radix-ui+react-toggle@1.1.9_@types+react-dom@19.1.6_@types+react@19.1.8__@types+react@19.1.8_qow4awvdp5vvbaipk7nw4amefq/node_modules/@radix-ui/react-toggle/dist/index.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/utils.ts [app-ssr] (ecmascript)");
'use client';
;
;
;
;
const toggleVariants = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$class$2d$variance$2d$authority$40$0$2e$7$2e$1$2f$node_modules$2f$class$2d$variance$2d$authority$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cva"])([
    'inline-flex items-center justify-center gap-2 rounded-full',
    'text-sm font-medium whitespace-nowrap',
    'cursor-pointer outline-none transition-[color,border,background-color]',
    'focus-visible:ring-ring/50 focus-visible:ring-[3px] focus-visible:border-ring',
    'aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive',
    'disabled:pointer-events-none disabled:opacity-50 disabled:not-allowed',
    'data-[state=on]:bg-button-selected data-[state=on]:border-button-border-selected',
    "[&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 [&_svg]:shrink-0"
], {
    variants: {
        variant: {
            default: 'bg-button hover:bg-muted focus:bg-muted hover:text-muted-foreground focus:text-muted-foreground',
            primary: 'text-fg1 bg-button hover:bg-button-hover focus:bg-button-hover data-[state=off]:bg-button-primary hover:data-[state=off]:bg-button-hover data-[state=off]:text-button-primary-foreground',
            secondary: 'text-fg1 bg-button hover:bg-button-hover focus:bg-button-hover data-[state=on]:bg-button-secondary hover:data-[state=on]:bg-button-secondary data-[state=on]:text-button-secondary-foreground',
            outline: [
                'border border-button-border bg-button text-button-foreground',
                'hover:bg-background focus:bg-background'
            ]
        },
        size: {
            default: 'h-9 px-2 min-w-9',
            sm: 'h-8 px-1.5 min-w-8',
            lg: 'h-10 px-2.5 min-w-10'
        }
    },
    defaultVariants: {
        variant: 'default',
        size: 'default'
    }
});
function Toggle({ className, variant, size, ...props }) {
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$toggle$40$1$2e$1$2e$9_$40$types$2b$react$2d$dom$40$19$2e$1$2e$6_$40$types$2b$react$40$19$2e$1$2e$8_$5f40$types$2b$react$40$19$2e$1$2e$8_qow4awvdp5vvbaipk7nw4amefq$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$toggle$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Root"], {
        "data-slot": "toggle",
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])(toggleVariants({
            variant,
            size,
            className
        })),
        ...props
    }, void 0, false, {
        fileName: "[project]/components/ui/toggle.tsx",
        lineNumber: 53,
        columnNumber: 5
    }, this);
}
;
}}),
"[project]/components/ui/select.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "Select": (()=>Select),
    "SelectContent": (()=>SelectContent),
    "SelectGroup": (()=>SelectGroup),
    "SelectItem": (()=>SelectItem),
    "SelectLabel": (()=>SelectLabel),
    "SelectScrollDownButton": (()=>SelectScrollDownButton),
    "SelectScrollUpButton": (()=>SelectScrollUpButton),
    "SelectSeparator": (()=>SelectSeparator),
    "SelectTrigger": (()=>SelectTrigger),
    "SelectValue": (()=>SelectValue)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$CaretDown$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@phosphor-icons+react@2.1.10_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/@phosphor-icons/react/dist/ssr/CaretDown.es.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$CaretUp$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@phosphor-icons+react@2.1.10_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/@phosphor-icons/react/dist/ssr/CaretUp.es.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$Check$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@phosphor-icons+react@2.1.10_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/@phosphor-icons/react/dist/ssr/Check.es.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$select$40$2$2e$2$2e$5_$40$types$2b$react$2d$dom$40$19$2e$1$2e$6_$40$types$2b$react$40$19$2e$1$2e$8_$5f40$types$2b$react$40$19$2e$1$2e$8_h3u75pabkk75sw2qjhdanlolna$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$select$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@radix-ui+react-select@2.2.5_@types+react-dom@19.1.6_@types+react@19.1.8__@types+react@19.1.8_h3u75pabkk75sw2qjhdanlolna/node_modules/@radix-ui/react-select/dist/index.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/utils.ts [app-ssr] (ecmascript)");
'use client';
;
;
;
;
function Select({ ...props }) {
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$select$40$2$2e$2$2e$5_$40$types$2b$react$2d$dom$40$19$2e$1$2e$6_$40$types$2b$react$40$19$2e$1$2e$8_$5f40$types$2b$react$40$19$2e$1$2e$8_h3u75pabkk75sw2qjhdanlolna$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$select$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Root"], {
        "data-slot": "select",
        ...props
    }, void 0, false, {
        fileName: "[project]/components/ui/select.tsx",
        lineNumber: 9,
        columnNumber: 10
    }, this);
}
function SelectGroup({ ...props }) {
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$select$40$2$2e$2$2e$5_$40$types$2b$react$2d$dom$40$19$2e$1$2e$6_$40$types$2b$react$40$19$2e$1$2e$8_$5f40$types$2b$react$40$19$2e$1$2e$8_h3u75pabkk75sw2qjhdanlolna$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$select$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Group"], {
        "data-slot": "select-group",
        ...props
    }, void 0, false, {
        fileName: "[project]/components/ui/select.tsx",
        lineNumber: 13,
        columnNumber: 10
    }, this);
}
function SelectValue({ ...props }) {
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$select$40$2$2e$2$2e$5_$40$types$2b$react$2d$dom$40$19$2e$1$2e$6_$40$types$2b$react$40$19$2e$1$2e$8_$5f40$types$2b$react$40$19$2e$1$2e$8_h3u75pabkk75sw2qjhdanlolna$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$select$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Value"], {
        "data-slot": "select-value",
        ...props
    }, void 0, false, {
        fileName: "[project]/components/ui/select.tsx",
        lineNumber: 17,
        columnNumber: 10
    }, this);
}
function SelectTrigger({ className, size = 'default', children, ...props }) {
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$select$40$2$2e$2$2e$5_$40$types$2b$react$2d$dom$40$19$2e$1$2e$6_$40$types$2b$react$40$19$2e$1$2e$8_$5f40$types$2b$react$40$19$2e$1$2e$8_h3u75pabkk75sw2qjhdanlolna$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$select$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Trigger"], {
        "data-slot": "select-trigger",
        "data-size": size,
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])([
            'flex w-fit cursor-pointer items-center justify-between gap-2',
            'rounded-full px-3 py-2 text-sm whitespace-nowrap',
            'bg-button transition-[color,border,background-color]',
            'disabled:cursor-not-allowed disabled:opacity-50',
            'hover:bg-muted focus:bg-muted hover:text-muted-foreground focus:text-muted-foreground focus-visible:border-ring focus-visible:ring-ring/50 outline-none focus-visible:ring-[3px]',
            'aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive',
            'data-[placeholder]:text-muted-foreground',
            'data-[size=default]:h-9 data-[size=sm]:h-8',
            '*:data-[slot=select-value]:line-clamp-1 *:data-[slot=select-value]:flex *:data-[slot=select-value]:items-center *:data-[slot=select-value]:gap-2',
            "[&:not([class*='text-'])_svg:not([class*='text-'])]:text-muted-foreground",
            "[&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4"
        ], className),
        ...props,
        children: [
            children,
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$select$40$2$2e$2$2e$5_$40$types$2b$react$2d$dom$40$19$2e$1$2e$6_$40$types$2b$react$40$19$2e$1$2e$8_$5f40$types$2b$react$40$19$2e$1$2e$8_h3u75pabkk75sw2qjhdanlolna$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$select$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Icon"], {
                asChild: true,
                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$CaretDown$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["CaretDownIcon"], {
                    weight: "bold",
                    className: "size-4 transition-colors"
                }, void 0, false, {
                    fileName: "[project]/components/ui/select.tsx",
                    lineNumber: 52,
                    columnNumber: 9
                }, this)
            }, void 0, false, {
                fileName: "[project]/components/ui/select.tsx",
                lineNumber: 51,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/components/ui/select.tsx",
        lineNumber: 29,
        columnNumber: 5
    }, this);
}
function SelectContent({ className, children, position = 'popper', ...props }) {
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$select$40$2$2e$2$2e$5_$40$types$2b$react$2d$dom$40$19$2e$1$2e$6_$40$types$2b$react$40$19$2e$1$2e$8_$5f40$types$2b$react$40$19$2e$1$2e$8_h3u75pabkk75sw2qjhdanlolna$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$select$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Portal"], {
        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$select$40$2$2e$2$2e$5_$40$types$2b$react$2d$dom$40$19$2e$1$2e$6_$40$types$2b$react$40$19$2e$1$2e$8_$5f40$types$2b$react$40$19$2e$1$2e$8_h3u75pabkk75sw2qjhdanlolna$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$select$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Content"], {
            "data-slot": "select-content",
            className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])([
                'bg-popover text-popover-foreground relative z-50 overflow-x-hidden overflow-y-auto rounded-md border drop-shadow-xl/5',
                'max-h-(--radix-select-content-available-height) min-w-[8rem]',
                'origin-(--radix-select-content-transform-origin)',
                'data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95',
                'data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95',
                'data-[side=top]:slide-in-from-bottom-2',
                'data-[side=right]:slide-in-from-left-2',
                'data-[side=bottom]:slide-in-from-top-2',
                'data-[side=left]:slide-in-from-right-2'
            ], position === 'popper' && 'data-[side=bottom]:translate-y-1 data-[side=left]:-translate-x-1 data-[side=right]:translate-x-1 data-[side=top]:-translate-y-1', className),
            position: position,
            ...props,
            children: [
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(SelectScrollUpButton, {}, void 0, false, {
                    fileName: "[project]/components/ui/select.tsx",
                    lineNumber: 87,
                    columnNumber: 9
                }, this),
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$select$40$2$2e$2$2e$5_$40$types$2b$react$2d$dom$40$19$2e$1$2e$6_$40$types$2b$react$40$19$2e$1$2e$8_$5f40$types$2b$react$40$19$2e$1$2e$8_h3u75pabkk75sw2qjhdanlolna$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$select$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Viewport"], {
                    className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])('p-1', position === 'popper' && 'h-[var(--radix-select-trigger-height)] w-full min-w-[var(--radix-select-trigger-width)] scroll-my-1'),
                    children: children
                }, void 0, false, {
                    fileName: "[project]/components/ui/select.tsx",
                    lineNumber: 88,
                    columnNumber: 9
                }, this),
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(SelectScrollDownButton, {}, void 0, false, {
                    fileName: "[project]/components/ui/select.tsx",
                    lineNumber: 97,
                    columnNumber: 9
                }, this)
            ]
        }, void 0, true, {
            fileName: "[project]/components/ui/select.tsx",
            lineNumber: 66,
            columnNumber: 7
        }, this)
    }, void 0, false, {
        fileName: "[project]/components/ui/select.tsx",
        lineNumber: 65,
        columnNumber: 5
    }, this);
}
function SelectLabel({ className, ...props }) {
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$select$40$2$2e$2$2e$5_$40$types$2b$react$2d$dom$40$19$2e$1$2e$6_$40$types$2b$react$40$19$2e$1$2e$8_$5f40$types$2b$react$40$19$2e$1$2e$8_h3u75pabkk75sw2qjhdanlolna$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$select$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Label"], {
        "data-slot": "select-label",
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])('text-muted-foreground px-2 py-1.5 text-xs', className),
        ...props
    }, void 0, false, {
        fileName: "[project]/components/ui/select.tsx",
        lineNumber: 105,
        columnNumber: 5
    }, this);
}
function SelectItem({ className, children, ...props }) {
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$select$40$2$2e$2$2e$5_$40$types$2b$react$2d$dom$40$19$2e$1$2e$6_$40$types$2b$react$40$19$2e$1$2e$8_$5f40$types$2b$react$40$19$2e$1$2e$8_h3u75pabkk75sw2qjhdanlolna$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$select$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Item"], {
        "data-slot": "select-item",
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])([
            'relative flex w-full cursor-default items-center gap-2 rounded-sm py-1.5 pr-8 pl-2 text-sm outline-hidden select-none',
            'cursor-pointer disabled:cursor-not-allowed',
            'focus:bg-accent focus:text-accent-foreground',
            'data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
            '*:[span]:last:flex *:[span]:last:items-center *:[span]:last:gap-2',
            "[&_svg:not([class*='text-'])]:text-muted-foreground [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4"
        ], className),
        ...props,
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                className: "absolute right-2 flex size-3.5 items-center justify-center",
                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$select$40$2$2e$2$2e$5_$40$types$2b$react$2d$dom$40$19$2e$1$2e$6_$40$types$2b$react$40$19$2e$1$2e$8_$5f40$types$2b$react$40$19$2e$1$2e$8_h3u75pabkk75sw2qjhdanlolna$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$select$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["ItemIndicator"], {
                    children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$Check$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["CheckIcon"], {
                        className: "size-4",
                        weight: "bold"
                    }, void 0, false, {
                        fileName: "[project]/components/ui/select.tsx",
                        lineNumber: 136,
                        columnNumber: 11
                    }, this)
                }, void 0, false, {
                    fileName: "[project]/components/ui/select.tsx",
                    lineNumber: 135,
                    columnNumber: 9
                }, this)
            }, void 0, false, {
                fileName: "[project]/components/ui/select.tsx",
                lineNumber: 134,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$select$40$2$2e$2$2e$5_$40$types$2b$react$2d$dom$40$19$2e$1$2e$6_$40$types$2b$react$40$19$2e$1$2e$8_$5f40$types$2b$react$40$19$2e$1$2e$8_h3u75pabkk75sw2qjhdanlolna$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$select$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["ItemText"], {
                children: children
            }, void 0, false, {
                fileName: "[project]/components/ui/select.tsx",
                lineNumber: 139,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/components/ui/select.tsx",
        lineNumber: 119,
        columnNumber: 5
    }, this);
}
function SelectSeparator({ className, ...props }) {
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$select$40$2$2e$2$2e$5_$40$types$2b$react$2d$dom$40$19$2e$1$2e$6_$40$types$2b$react$40$19$2e$1$2e$8_$5f40$types$2b$react$40$19$2e$1$2e$8_h3u75pabkk75sw2qjhdanlolna$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$select$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Separator"], {
        "data-slot": "select-separator",
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])('bg-border pointer-events-none -mx-1 my-1 h-px', className),
        ...props
    }, void 0, false, {
        fileName: "[project]/components/ui/select.tsx",
        lineNumber: 149,
        columnNumber: 5
    }, this);
}
function SelectScrollUpButton({ className, ...props }) {
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$select$40$2$2e$2$2e$5_$40$types$2b$react$2d$dom$40$19$2e$1$2e$6_$40$types$2b$react$40$19$2e$1$2e$8_$5f40$types$2b$react$40$19$2e$1$2e$8_h3u75pabkk75sw2qjhdanlolna$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$select$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["ScrollUpButton"], {
        "data-slot": "select-scroll-up-button",
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])('flex cursor-default items-center justify-center py-1', className),
        ...props,
        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$CaretUp$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["CaretUpIcon"], {
            className: "size-4",
            weight: "bold"
        }, void 0, false, {
            fileName: "[project]/components/ui/select.tsx",
            lineNumber: 167,
            columnNumber: 7
        }, this)
    }, void 0, false, {
        fileName: "[project]/components/ui/select.tsx",
        lineNumber: 162,
        columnNumber: 5
    }, this);
}
function SelectScrollDownButton({ className, ...props }) {
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$radix$2d$ui$2b$react$2d$select$40$2$2e$2$2e$5_$40$types$2b$react$2d$dom$40$19$2e$1$2e$6_$40$types$2b$react$40$19$2e$1$2e$8_$5f40$types$2b$react$40$19$2e$1$2e$8_h3u75pabkk75sw2qjhdanlolna$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$select$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["ScrollDownButton"], {
        "data-slot": "select-scroll-down-button",
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])('flex cursor-default items-center justify-center py-1', className),
        ...props,
        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$CaretDown$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["CaretDownIcon"], {
            className: "size-4",
            weight: "bold"
        }, void 0, false, {
            fileName: "[project]/components/ui/select.tsx",
            lineNumber: 182,
            columnNumber: 7
        }, this)
    }, void 0, false, {
        fileName: "[project]/components/ui/select.tsx",
        lineNumber: 177,
        columnNumber: 5
    }, this);
}
;
}}),
"[project]/components/livekit/device-select.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "DeviceSelect": (()=>DeviceSelect)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$class$2d$variance$2d$authority$40$0$2e$7$2e$1$2f$node_modules$2f$class$2d$variance$2d$authority$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/class-variance-authority@0.7.1/node_modules/class-variance-authority/dist/index.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__w__as__useMaybeRoomContext$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/contexts-CH4CA2DN.mjs [app-ssr] (ecmascript) <export w as useMaybeRoomContext>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__e__as__useMediaDeviceSelect$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-ssr] (ecmascript) <export e as useMediaDeviceSelect>");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$select$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/ui/select.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/utils.ts [app-ssr] (ecmascript)");
'use client';
;
;
;
;
;
const selectVariants = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$class$2d$variance$2d$authority$40$0$2e$7$2e$1$2f$node_modules$2f$class$2d$variance$2d$authority$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cva"])([
    'w-full rounded-full px-3 py-2 text-sm cursor-pointer',
    'disabled:not-allowed hover:bg-button-hover focus:bg-button-hover'
], {
    variants: {
        size: {
            default: 'w-[180px]',
            sm: 'w-auto'
        }
    },
    defaultVariants: {
        size: 'default'
    }
});
function DeviceSelect({ kind, track, requestPermissions, onError, // initialSelection,
// onActiveDeviceChange,
// onDeviceListChange,
...props }) {
    const size = props.size || 'default';
    const room = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__w__as__useMaybeRoomContext$3e$__["useMaybeRoomContext"])();
    const { devices, activeDeviceId, setActiveMediaDevice } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__e__as__useMediaDeviceSelect$3e$__["useMediaDeviceSelect"])({
        kind,
        room,
        track,
        requestPermissions,
        onError
    });
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$select$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Select"], {
        value: activeDeviceId,
        onValueChange: setActiveMediaDevice,
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$select$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["SelectTrigger"], {
                className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])(selectVariants({
                    size
                }), props.className),
                children: size !== 'sm' && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$select$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["SelectValue"], {
                    className: "font-mono text-sm",
                    placeholder: `Select a ${kind}`
                }, void 0, false, {
                    fileName: "[project]/components/livekit/device-select.tsx",
                    lineNumber: 68,
                    columnNumber: 11
                }, this)
            }, void 0, false, {
                fileName: "[project]/components/livekit/device-select.tsx",
                lineNumber: 66,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$select$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["SelectContent"], {
                children: devices.map((device)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$select$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["SelectItem"], {
                        value: device.deviceId,
                        className: "font-mono text-xs",
                        children: device.label
                    }, device.deviceId, false, {
                        fileName: "[project]/components/livekit/device-select.tsx",
                        lineNumber: 73,
                        columnNumber: 11
                    }, this))
            }, void 0, false, {
                fileName: "[project]/components/livekit/device-select.tsx",
                lineNumber: 71,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/components/livekit/device-select.tsx",
        lineNumber: 65,
        columnNumber: 5
    }, this);
}
}}),
"[project]/components/livekit/track-toggle.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "TrackToggle": (()=>TrackToggle)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-client@2.15.2_@types+dom-mediacapture-record@1.0.22/node_modules/livekit-client/dist/livekit-client.esm.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$Microphone$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@phosphor-icons+react@2.1.10_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/@phosphor-icons/react/dist/ssr/Microphone.es.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$MicrophoneSlash$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@phosphor-icons+react@2.1.10_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/@phosphor-icons/react/dist/ssr/MicrophoneSlash.es.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$MonitorArrowUp$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@phosphor-icons+react@2.1.10_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/@phosphor-icons/react/dist/ssr/MonitorArrowUp.es.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$Spinner$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@phosphor-icons+react@2.1.10_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/@phosphor-icons/react/dist/ssr/Spinner.es.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$VideoCamera$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@phosphor-icons+react@2.1.10_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/@phosphor-icons/react/dist/ssr/VideoCamera.es.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$VideoCameraSlash$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@phosphor-icons+react@2.1.10_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/@phosphor-icons/react/dist/ssr/VideoCameraSlash.es.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$toggle$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/ui/toggle.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/utils.ts [app-ssr] (ecmascript)");
'use client';
;
;
;
;
;
;
function getSourceIcon(source, enabled, pending = false) {
    if (pending) {
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$Spinner$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["SpinnerIcon"];
    }
    switch(source){
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.Microphone:
            return enabled ? __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$Microphone$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["MicrophoneIcon"] : __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$MicrophoneSlash$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["MicrophoneSlashIcon"];
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.Camera:
            return enabled ? __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$VideoCamera$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["VideoCameraIcon"] : __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$VideoCameraSlash$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["VideoCameraSlashIcon"];
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.ScreenShare:
            return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$MonitorArrowUp$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["MonitorArrowUpIcon"];
        default:
            return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Fragment"];
    }
}
function TrackToggle({ source, pressed, pending, className, ...props }) {
    const IconComponent = getSourceIcon(source, pressed ?? false, pending);
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$toggle$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Toggle"], {
        pressed: pressed,
        "aria-label": `Toggle ${source}`,
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])(className),
        ...props,
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(IconComponent, {
                weight: "bold",
                className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])(pending && 'animate-spin')
            }, void 0, false, {
                fileName: "[project]/components/livekit/track-toggle.tsx",
                lineNumber: 44,
                columnNumber: 7
            }, this),
            props.children
        ]
    }, void 0, true, {
        fileName: "[project]/components/livekit/track-toggle.tsx",
        lineNumber: 43,
        columnNumber: 5
    }, this);
}
}}),
"[project]/components/livekit/agent-control-bar/hooks/use-publish-permissions.ts [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "usePublishPermissions": (()=>usePublishPermissions)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-client@2.15.2_@types+dom-mediacapture-record@1.0.22/node_modules/livekit-client/dist/livekit-client.esm.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__A__as__useLocalParticipantPermissions$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-ssr] (ecmascript) <export A as useLocalParticipantPermissions>");
;
;
const trackSourceToProtocol = (source)=>{
    // NOTE: this mapping avoids importing the protocol package as that leads to a significant bundle size increase
    switch(source){
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.Camera:
            return 1;
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.Microphone:
            return 2;
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.ScreenShare:
            return 3;
        default:
            return 0;
    }
};
function usePublishPermissions() {
    const localPermissions = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__A__as__useLocalParticipantPermissions$3e$__["useLocalParticipantPermissions"])();
    const canPublishSource = (source)=>{
        return !!localPermissions?.canPublish && (localPermissions.canPublishSources.length === 0 || localPermissions.canPublishSources.includes(trackSourceToProtocol(source)));
    };
    return {
        camera: canPublishSource(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.Camera),
        microphone: canPublishSource(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.Microphone),
        screenShare: canPublishSource(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.ScreenShare),
        data: localPermissions?.canPublishData ?? false
    };
}
}}),
"[project]/components/livekit/agent-control-bar/hooks/use-agent-control-bar.ts [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "useAgentControlBar": (()=>useAgentControlBar)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-client@2.15.2_@types+dom-mediacapture-record@1.0.22/node_modules/livekit-client/dist/livekit-client.esm.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__C__as__useLocalParticipant$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-ssr] (ecmascript) <export C as useLocalParticipant>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__x__as__usePersistentUserChoices$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-ssr] (ecmascript) <export x as usePersistentUserChoices>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__f__as__useRoomContext$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/contexts-CH4CA2DN.mjs [app-ssr] (ecmascript) <export f as useRoomContext>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__h__as__useTrackToggle$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-ssr] (ecmascript) <export h as useTrackToggle>");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$agent$2d$control$2d$bar$2f$hooks$2f$use$2d$publish$2d$permissions$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/livekit/agent-control-bar/hooks/use-publish-permissions.ts [app-ssr] (ecmascript)");
;
;
;
;
function useAgentControlBar(props = {}) {
    const { controls, saveUserChoices = true } = props;
    const visibleControls = {
        leave: true,
        ...controls
    };
    const { microphoneTrack, localParticipant } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__C__as__useLocalParticipant$3e$__["useLocalParticipant"])();
    const publishPermissions = (0, __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$agent$2d$control$2d$bar$2f$hooks$2f$use$2d$publish$2d$permissions$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["usePublishPermissions"])();
    const room = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__f__as__useRoomContext$3e$__["useRoomContext"])();
    const microphoneToggle = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__h__as__useTrackToggle$3e$__["useTrackToggle"])({
        source: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.Microphone,
        onDeviceError: (error)=>props.onDeviceError?.({
                source: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.Microphone,
                error
            })
    });
    const cameraToggle = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__h__as__useTrackToggle$3e$__["useTrackToggle"])({
        source: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.Camera,
        onDeviceError: (error)=>props.onDeviceError?.({
                source: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.Camera,
                error
            })
    });
    const screenShareToggle = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__h__as__useTrackToggle$3e$__["useTrackToggle"])({
        source: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.ScreenShare,
        onDeviceError: (error)=>props.onDeviceError?.({
                source: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.ScreenShare,
                error
            })
    });
    const micTrackRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useMemo"])(()=>{
        return {
            participant: localParticipant,
            source: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.Microphone,
            publication: microphoneTrack
        };
    }, [
        localParticipant,
        microphoneTrack
    ]);
    visibleControls.microphone ??= publishPermissions.microphone;
    visibleControls.screenShare ??= publishPermissions.screenShare;
    visibleControls.camera ??= publishPermissions.camera;
    visibleControls.chat ??= publishPermissions.data;
    const { saveAudioInputEnabled, saveAudioInputDeviceId, saveVideoInputEnabled, saveVideoInputDeviceId } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__x__as__usePersistentUserChoices$3e$__["usePersistentUserChoices"])({
        preventSave: !saveUserChoices
    });
    const handleDisconnect = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useCallback"])(()=>{
        if (room) {
            room.disconnect();
        }
    }, [
        room
    ]);
    const handleAudioDeviceChange = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useCallback"])((deviceId)=>{
        saveAudioInputDeviceId(deviceId ?? 'default');
    }, [
        saveAudioInputDeviceId
    ]);
    const handleVideoDeviceChange = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useCallback"])((deviceId)=>{
        saveVideoInputDeviceId(deviceId ?? 'default');
    }, [
        saveVideoInputDeviceId
    ]);
    const handleToggleCamera = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useCallback"])(async (enabled)=>{
        if (screenShareToggle.enabled) {
            screenShareToggle.toggle(false);
        }
        await cameraToggle.toggle(enabled);
        // persist video input enabled preference
        saveVideoInputEnabled(!cameraToggle.enabled);
    }, [
        cameraToggle.enabled,
        screenShareToggle.enabled
    ]);
    const handleToggleMicrophone = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useCallback"])(async (enabled)=>{
        await microphoneToggle.toggle(enabled);
        // persist audio input enabled preference
        saveAudioInputEnabled(!microphoneToggle.enabled);
    }, [
        microphoneToggle.enabled
    ]);
    const handleToggleScreenShare = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useCallback"])(async (enabled)=>{
        if (cameraToggle.enabled) {
            cameraToggle.toggle(false);
        }
        await screenShareToggle.toggle(enabled);
    }, [
        screenShareToggle.enabled,
        cameraToggle.enabled
    ]);
    return {
        micTrackRef,
        visibleControls,
        cameraToggle: {
            ...cameraToggle,
            toggle: handleToggleCamera
        },
        microphoneToggle: {
            ...microphoneToggle,
            toggle: handleToggleMicrophone
        },
        screenShareToggle: {
            ...screenShareToggle,
            toggle: handleToggleScreenShare
        },
        handleDisconnect,
        handleAudioDeviceChange,
        handleVideoDeviceChange
    };
}
}}),
"[project]/components/livekit/agent-control-bar/agent-control-bar.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "AgentControlBar": (()=>AgentControlBar)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-client@2.15.2_@types+dom-mediacapture-record@1.0.22/node_modules/livekit-client/dist/livekit-client.esm.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__B__as__BarVisualizer$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/components-k0KtCs0w.mjs [app-ssr] (ecmascript) <export B as BarVisualizer>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__M__as__useRemoteParticipants$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-ssr] (ecmascript) <export M as useRemoteParticipants>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$ChatText$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@phosphor-icons+react@2.1.10_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/@phosphor-icons/react/dist/ssr/ChatText.es.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$PhoneDisconnect$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@phosphor-icons+react@2.1.10_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/@phosphor-icons/react/dist/ssr/PhoneDisconnect.es.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$chat$2f$chat$2d$input$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/livekit/chat/chat-input.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$button$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/ui/button.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$toggle$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/ui/toggle.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/utils.ts [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$device$2d$select$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/livekit/device-select.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$track$2d$toggle$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/livekit/track-toggle.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$agent$2d$control$2d$bar$2f$hooks$2f$use$2d$agent$2d$control$2d$bar$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/livekit/agent-control-bar/hooks/use-agent-control-bar.ts [app-ssr] (ecmascript)");
'use client';
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
;
;
function AgentControlBar({ controls, saveUserChoices = true, capabilities, className, onSendMessage, onChatOpenChange, onDisconnect, onDeviceError, ...props }) {
    const participants = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__M__as__useRemoteParticipants$3e$__["useRemoteParticipants"])();
    const [chatOpen, setChatOpen] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(false);
    const [isSendingMessage, setIsSendingMessage] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(false);
    const isAgentAvailable = participants.some((p)=>p.isAgent);
    const isInputDisabled = !chatOpen || !isAgentAvailable || isSendingMessage;
    const { micTrackRef, visibleControls, cameraToggle, microphoneToggle, screenShareToggle, handleAudioDeviceChange, handleVideoDeviceChange, handleDisconnect } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$agent$2d$control$2d$bar$2f$hooks$2f$use$2d$agent$2d$control$2d$bar$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useAgentControlBar"])({
        controls,
        saveUserChoices
    });
    const handleSendMessage = async (message)=>{
        setIsSendingMessage(true);
        try {
            await onSendMessage?.(message);
        } finally{
            setIsSendingMessage(false);
        }
    };
    const onLeave = ()=>{
        handleDisconnect();
        onDisconnect?.();
    };
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        onChatOpenChange?.(chatOpen);
    }, [
        chatOpen,
        onChatOpenChange
    ]);
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        "aria-label": "Voice assistant controls",
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])('bg-background border-bg2 dark:border-separator1 flex flex-col rounded-[31px] border p-3 drop-shadow-md/3', className),
        ...props,
        children: [
            capabilities.supportsChatInput && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                inert: !chatOpen,
                className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])('overflow-hidden transition-[height] duration-300 ease-out', chatOpen ? 'h-[57px]' : 'h-0'),
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex h-8 w-full",
                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$chat$2f$chat$2d$input$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["ChatInput"], {
                            onSend: handleSendMessage,
                            disabled: isInputDisabled,
                            className: "w-full"
                        }, void 0, false, {
                            fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                            lineNumber: 97,
                            columnNumber: 13
                        }, this)
                    }, void 0, false, {
                        fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                        lineNumber: 96,
                        columnNumber: 11
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("hr", {
                        className: "border-bg2 my-3"
                    }, void 0, false, {
                        fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                        lineNumber: 99,
                        columnNumber: 11
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                lineNumber: 89,
                columnNumber: 9
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "flex flex-row justify-between gap-1",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex gap-1",
                        children: [
                            visibleControls.microphone && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex items-center gap-0",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$track$2d$toggle$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["TrackToggle"], {
                                        variant: "primary",
                                        source: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.Microphone,
                                        pressed: microphoneToggle.enabled,
                                        disabled: microphoneToggle.pending,
                                        onPressedChange: microphoneToggle.toggle,
                                        className: "peer/track group/track relative w-auto pr-3 pl-3 md:rounded-r-none md:border-r-0 md:pr-2",
                                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__B__as__BarVisualizer$3e$__["BarVisualizer"], {
                                            barCount: 3,
                                            trackRef: micTrackRef,
                                            options: {
                                                minHeight: 5
                                            },
                                            className: "flex h-full w-auto items-center justify-center gap-0.5",
                                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])([
                                                    'h-full w-0.5 origin-center rounded-2xl',
                                                    'group-data-[state=on]/track:bg-fg1 group-data-[state=off]/track:bg-destructive-foreground',
                                                    'data-lk-muted:bg-muted'
                                                ])
                                            }, void 0, false, {
                                                fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                                                lineNumber: 121,
                                                columnNumber: 19
                                            }, this)
                                        }, void 0, false, {
                                            fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                                            lineNumber: 115,
                                            columnNumber: 17
                                        }, this)
                                    }, void 0, false, {
                                        fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                                        lineNumber: 107,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("hr", {
                                        className: "bg-separator1 peer-data-[state=off]/track:bg-separatorSerious relative z-10 -mr-px hidden h-4 w-px md:block"
                                    }, void 0, false, {
                                        fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                                        lineNumber: 130,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$device$2d$select$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["DeviceSelect"], {
                                        size: "sm",
                                        kind: "audioinput",
                                        onError: (error)=>onDeviceError?.({
                                                source: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.Microphone,
                                                error: error
                                            }),
                                        onActiveDeviceChange: handleAudioDeviceChange,
                                        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])([
                                            'pl-2',
                                            'peer-data-[state=off]/track:text-destructive-foreground',
                                            'hover:text-fg1 focus:text-fg1',
                                            'hover:peer-data-[state=off]/track:text-destructive-foreground focus:peer-data-[state=off]/track:text-destructive-foreground',
                                            'hidden rounded-l-none md:block'
                                        ])
                                    }, void 0, false, {
                                        fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                                        lineNumber: 131,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                                lineNumber: 106,
                                columnNumber: 13
                            }, this),
                            capabilities.supportsVideoInput && visibleControls.camera && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex items-center gap-0",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$track$2d$toggle$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["TrackToggle"], {
                                        variant: "primary",
                                        source: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.Camera,
                                        pressed: cameraToggle.enabled,
                                        pending: cameraToggle.pending,
                                        disabled: cameraToggle.pending,
                                        onPressedChange: cameraToggle.toggle,
                                        className: "peer/track relative w-auto rounded-r-none pr-3 pl-3 disabled:opacity-100 md:border-r-0 md:pr-2"
                                    }, void 0, false, {
                                        fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                                        lineNumber: 151,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("hr", {
                                        className: "bg-separator1 peer-data-[state=off]/track:bg-separatorSerious relative z-10 -mr-px hidden h-4 w-px md:block"
                                    }, void 0, false, {
                                        fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                                        lineNumber: 160,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$device$2d$select$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["DeviceSelect"], {
                                        size: "sm",
                                        kind: "videoinput",
                                        onError: (error)=>onDeviceError?.({
                                                source: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.Camera,
                                                error: error
                                            }),
                                        onActiveDeviceChange: handleVideoDeviceChange,
                                        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])([
                                            'pl-2',
                                            'peer-data-[state=off]/track:text-destructive-foreground',
                                            'hover:text-fg1 focus:text-fg1',
                                            'hover:peer-data-[state=off]/track:text-destructive-foreground focus:peer-data-[state=off]/track:text-destructive-foreground',
                                            'rounded-l-none'
                                        ])
                                    }, void 0, false, {
                                        fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                                        lineNumber: 161,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                                lineNumber: 150,
                                columnNumber: 13
                            }, this),
                            capabilities.supportsScreenShare && visibleControls.screenShare && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex items-center gap-0",
                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$track$2d$toggle$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["TrackToggle"], {
                                    variant: "secondary",
                                    source: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.ScreenShare,
                                    pressed: screenShareToggle.enabled,
                                    disabled: screenShareToggle.pending,
                                    onPressedChange: screenShareToggle.toggle,
                                    className: "relative w-auto"
                                }, void 0, false, {
                                    fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                                    lineNumber: 181,
                                    columnNumber: 15
                                }, this)
                            }, void 0, false, {
                                fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                                lineNumber: 180,
                                columnNumber: 13
                            }, this),
                            visibleControls.chat && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$toggle$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Toggle"], {
                                variant: "secondary",
                                "aria-label": "Toggle chat",
                                pressed: chatOpen,
                                onPressedChange: setChatOpen,
                                disabled: !isAgentAvailable,
                                className: "aspect-square h-full",
                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$ChatText$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["ChatTextIcon"], {
                                    weight: "bold"
                                }, void 0, false, {
                                    fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                                    lineNumber: 201,
                                    columnNumber: 15
                                }, this)
                            }, void 0, false, {
                                fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                                lineNumber: 193,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                        lineNumber: 104,
                        columnNumber: 9
                    }, this),
                    visibleControls.leave && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$button$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Button"], {
                        variant: "destructive",
                        onClick: onLeave,
                        className: "font-mono",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$PhoneDisconnect$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["PhoneDisconnectIcon"], {
                                weight: "bold"
                            }, void 0, false, {
                                fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                                lineNumber: 207,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                className: "hidden md:inline",
                                children: "END CALL"
                            }, void 0, false, {
                                fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                                lineNumber: 208,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                className: "inline md:hidden",
                                children: "END"
                            }, void 0, false, {
                                fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                                lineNumber: 209,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                        lineNumber: 206,
                        columnNumber: 11
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
                lineNumber: 103,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/components/livekit/agent-control-bar/agent-control-bar.tsx",
        lineNumber: 80,
        columnNumber: 5
    }, this);
}
}}),
"[project]/components/livekit/chat/hooks/utils.ts [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "useChatMessage": (()=>useChatMessage)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
;
const useChatMessage = (entry, messageFormatter)=>{
    const formattedMessage = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useMemo"])(()=>{
        return messageFormatter ? messageFormatter(entry.message) : entry.message;
    }, [
        entry.message,
        messageFormatter
    ]);
    const hasBeenEdited = !!entry.editTimestamp;
    const time = new Date(entry.timestamp);
    const locale = typeof navigator !== 'undefined' ? navigator.language : 'en-US';
    const name = entry.from?.name && entry.from.name !== '' ? entry.from.name : entry.from?.identity;
    return {
        message: formattedMessage,
        hasBeenEdited,
        time,
        locale,
        name
    };
};
}}),
"[project]/components/livekit/chat/chat-entry.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "ChatEntry": (()=>ChatEntry)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/utils.ts [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$chat$2f$hooks$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/livekit/chat/hooks/utils.ts [app-ssr] (ecmascript)");
;
;
;
const ChatEntry = ({ entry, messageFormatter, hideName, hideTimestamp, className, ...props })=>{
    const { message, hasBeenEdited, time, locale, name } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$chat$2f$hooks$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useChatMessage"])(entry, messageFormatter);
    const isUser = entry.from?.isLocal ?? false;
    const messageOrigin = isUser ? 'remote' : 'local';
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("li", {
        "data-lk-message-origin": messageOrigin,
        title: time.toLocaleTimeString(locale, {
            timeStyle: 'full'
        }),
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])('group flex flex-col gap-0.5', className),
        ...props,
        children: [
            (!hideTimestamp || !hideName || hasBeenEdited) && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                className: "text-muted-foreground flex text-sm",
                children: [
                    !hideName && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("strong", {
                        className: "mt-2",
                        children: name
                    }, void 0, false, {
                        fileName: "[project]/components/livekit/chat/chat-entry.tsx",
                        lineNumber: 39,
                        columnNumber: 25
                    }, this),
                    !hideTimestamp && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                        className: "align-self-end ml-auto font-mono text-xs opacity-0 transition-opacity ease-linear group-hover:opacity-100",
                        children: [
                            hasBeenEdited && '*',
                            time.toLocaleTimeString(locale, {
                                timeStyle: 'short'
                            })
                        ]
                    }, void 0, true, {
                        fileName: "[project]/components/livekit/chat/chat-entry.tsx",
                        lineNumber: 42,
                        columnNumber: 13
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/components/livekit/chat/chat-entry.tsx",
                lineNumber: 38,
                columnNumber: 9
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])('max-w-4/5 rounded-[20px] p-2', isUser ? 'bg-muted ml-auto' : 'mr-auto'),
                children: message
            }, void 0, false, {
                fileName: "[project]/components/livekit/chat/chat-entry.tsx",
                lineNumber: 50,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/components/livekit/chat/chat-entry.tsx",
        lineNumber: 31,
        columnNumber: 5
    }, this);
};
}}),
"[project]/components/livekit/chat/chat-message-view.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "ChatMessageView": (()=>ChatMessageView),
    "useAutoScroll": (()=>useAutoScroll)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/utils.ts [app-ssr] (ecmascript)");
'use client';
;
;
;
function useAutoScroll(scrollContentContainerRef) {
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        function scrollToBottom() {
            const { scrollingElement } = document;
            if (scrollingElement) {
                scrollingElement.scrollTop = scrollingElement.scrollHeight;
            }
        }
        if (scrollContentContainerRef.current) {
            const resizeObserver = new ResizeObserver(scrollToBottom);
            resizeObserver.observe(scrollContentContainerRef.current);
            scrollToBottom();
            return ()=>resizeObserver.disconnect();
        }
    }, [
        scrollContentContainerRef
    ]);
}
const ChatMessageView = ({ className, children, ...props })=>{
    const scrollContentRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useRef"])(null);
    useAutoScroll(scrollContentRef);
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        ref: scrollContentRef,
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])('flex flex-col justify-end', className),
        ...props,
        children: children
    }, void 0, false, {
        fileName: "[project]/components/livekit/chat/chat-message-view.tsx",
        lineNumber: 37,
        columnNumber: 5
    }, this);
};
}}),
"[project]/components/livekit/agent-tile.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "AgentTile": (()=>AgentTile)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__B__as__BarVisualizer$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/components-k0KtCs0w.mjs [app-ssr] (ecmascript) <export B as BarVisualizer>");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/utils.ts [app-ssr] (ecmascript)");
;
;
;
const AgentTile = ({ state, audioTrack, className, ref })=>{
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        ref: ref,
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])(className),
        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__B__as__BarVisualizer$3e$__["BarVisualizer"], {
            barCount: 5,
            state: state,
            options: {
                minHeight: 5
            },
            trackRef: audioTrack,
            className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])('flex aspect-video w-40 items-center justify-center gap-1'),
            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])([
                    'bg-muted min-h-4 w-4 rounded-full',
                    'origin-center transition-colors duration-250 ease-linear',
                    'data-[lk-highlighted=true]:bg-foreground data-[lk-muted=true]:bg-muted'
                ])
            }, void 0, false, {
                fileName: "[project]/components/livekit/agent-tile.tsx",
                lineNumber: 25,
                columnNumber: 9
            }, this)
        }, void 0, false, {
            fileName: "[project]/components/livekit/agent-tile.tsx",
            lineNumber: 18,
            columnNumber: 7
        }, this)
    }, void 0, false, {
        fileName: "[project]/components/livekit/agent-tile.tsx",
        lineNumber: 17,
        columnNumber: 5
    }, this);
};
}}),
"[project]/components/livekit/avatar-tile.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "AvatarTile": (()=>AvatarTile)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__V__as__VideoTrack$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/components-k0KtCs0w.mjs [app-ssr] (ecmascript) <export V as VideoTrack>");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/utils.ts [app-ssr] (ecmascript)");
;
;
;
const AvatarTile = ({ videoTrack, className, ref })=>{
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        ref: ref,
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])(className),
        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__V__as__VideoTrack$3e$__["VideoTrack"], {
            trackRef: videoTrack,
            width: videoTrack?.publication.dimensions?.width ?? 0,
            height: videoTrack?.publication.dimensions?.height ?? 0,
            className: "rounded-md"
        }, void 0, false, {
            fileName: "[project]/components/livekit/avatar-tile.tsx",
            lineNumber: 16,
            columnNumber: 7
        }, this)
    }, void 0, false, {
        fileName: "[project]/components/livekit/avatar-tile.tsx",
        lineNumber: 15,
        columnNumber: 5
    }, this);
};
}}),
"[project]/components/livekit/video-tile.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "VideoTile": (()=>VideoTile)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$motion$40$12$2e$16$2e$0_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$motion$2f$dist$2f$es$2f$framer$2d$motion$2f$dist$2f$es$2f$render$2f$components$2f$motion$2f$proxy$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/motion@12.16.0_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/motion/dist/es/framer-motion/dist/es/render/components/motion/proxy.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__V__as__VideoTrack$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/components-k0KtCs0w.mjs [app-ssr] (ecmascript) <export V as VideoTrack>");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/utils.ts [app-ssr] (ecmascript)");
;
;
;
;
const MotionVideoTrack = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$motion$40$12$2e$16$2e$0_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$motion$2f$dist$2f$es$2f$framer$2d$motion$2f$dist$2f$es$2f$render$2f$components$2f$motion$2f$proxy$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["motion"].create(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__V__as__VideoTrack$3e$__["VideoTrack"]);
const VideoTile = ({ trackRef, className, ref })=>{
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        ref: ref,
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])('bg-muted overflow-hidden rounded-md', className),
        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(MotionVideoTrack, {
            trackRef: trackRef,
            width: trackRef?.publication.dimensions?.width ?? 0,
            height: trackRef?.publication.dimensions?.height ?? 0,
            className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])('h-full w-auto')
        }, void 0, false, {
            fileName: "[project]/components/livekit/video-tile.tsx",
            lineNumber: 15,
            columnNumber: 7
        }, this)
    }, void 0, false, {
        fileName: "[project]/components/livekit/video-tile.tsx",
        lineNumber: 14,
        columnNumber: 5
    }, this);
};
}}),
"[project]/components/livekit/media-tiles.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "MediaTiles": (()=>MediaTiles),
    "useLocalTrackRef": (()=>useLocalTrackRef)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-client@2.15.2_@types+dom-mediacapture-record@1.0.22/node_modules/livekit-client/dist/livekit-client.esm.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$motion$40$12$2e$16$2e$0_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$motion$2f$dist$2f$es$2f$framer$2d$motion$2f$dist$2f$es$2f$components$2f$AnimatePresence$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/motion@12.16.0_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/motion/dist/es/framer-motion/dist/es/components/AnimatePresence/index.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$motion$40$12$2e$16$2e$0_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$motion$2f$dist$2f$es$2f$framer$2d$motion$2f$dist$2f$es$2f$render$2f$components$2f$motion$2f$proxy$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/motion@12.16.0_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/motion/dist/es/framer-motion/dist/es/render/components/motion/proxy.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__C__as__useLocalParticipant$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-ssr] (ecmascript) <export C as useLocalParticipant>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__t__as__useTracks$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-ssr] (ecmascript) <export t as useTracks>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__V__as__useVoiceAssistant$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-ssr] (ecmascript) <export V as useVoiceAssistant>");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/utils.ts [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$agent$2d$tile$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/livekit/agent-tile.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$avatar$2d$tile$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/livekit/avatar-tile.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$video$2d$tile$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/livekit/video-tile.tsx [app-ssr] (ecmascript)");
;
;
;
;
;
;
;
;
;
const MotionVideoTile = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$motion$40$12$2e$16$2e$0_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$motion$2f$dist$2f$es$2f$framer$2d$motion$2f$dist$2f$es$2f$render$2f$components$2f$motion$2f$proxy$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["motion"].create(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$video$2d$tile$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["VideoTile"]);
const MotionAgentTile = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$motion$40$12$2e$16$2e$0_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$motion$2f$dist$2f$es$2f$framer$2d$motion$2f$dist$2f$es$2f$render$2f$components$2f$motion$2f$proxy$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["motion"].create(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$agent$2d$tile$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["AgentTile"]);
const MotionAvatarTile = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$motion$40$12$2e$16$2e$0_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$motion$2f$dist$2f$es$2f$framer$2d$motion$2f$dist$2f$es$2f$render$2f$components$2f$motion$2f$proxy$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["motion"].create(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$avatar$2d$tile$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["AvatarTile"]);
const animationProps = {
    initial: {
        opacity: 0,
        scale: 0
    },
    animate: {
        opacity: 1,
        scale: 1
    },
    exit: {
        opacity: 0,
        scale: 0
    },
    transition: {
        type: 'spring',
        stiffness: 675,
        damping: 75,
        mass: 1
    }
};
const classNames = {
    // GRID
    // 2 Columns x 3 Rows
    grid: [
        'h-full w-full',
        'grid gap-x-2 place-content-center',
        'grid-cols-[1fr_1fr] grid-rows-[90px_1fr_90px]'
    ],
    // Agent
    // chatOpen: true,
    // hasSecondTile: true
    // layout: Column 1 / Row 1
    // align: x-end y-center
    agentChatOpenWithSecondTile: [
        'col-start-1 row-start-1',
        'self-center justify-self-end'
    ],
    // Agent
    // chatOpen: true,
    // hasSecondTile: false
    // layout: Column 1 / Row 1 / Column-Span 2
    // align: x-center y-center
    agentChatOpenWithoutSecondTile: [
        'col-start-1 row-start-1',
        'col-span-2',
        'place-content-center'
    ],
    // Agent
    // chatOpen: false
    // layout: Column 1 / Row 1 / Column-Span 2 / Row-Span 3
    // align: x-center y-center
    agentChatClosed: [
        'col-start-1 row-start-1',
        'col-span-2 row-span-3',
        'place-content-center'
    ],
    // Second tile
    // chatOpen: true,
    // hasSecondTile: true
    // layout: Column 2 / Row 1
    // align: x-start y-center
    secondTileChatOpen: [
        'col-start-2 row-start-1',
        'self-center justify-self-start'
    ],
    // Second tile
    // chatOpen: false,
    // hasSecondTile: false
    // layout: Column 2 / Row 2
    // align: x-end y-end
    secondTileChatClosed: [
        'col-start-2 row-start-3',
        'place-content-end'
    ]
};
function useLocalTrackRef(source) {
    const { localParticipant } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__C__as__useLocalParticipant$3e$__["useLocalParticipant"])();
    const publication = localParticipant.getTrackPublication(source);
    const trackRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useMemo"])(()=>publication ? {
            source,
            participant: localParticipant,
            publication
        } : undefined, [
        source,
        publication,
        localParticipant
    ]);
    return trackRef;
}
function MediaTiles({ chatOpen }) {
    const { state: agentState, audioTrack: agentAudioTrack, videoTrack: agentVideoTrack } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__V__as__useVoiceAssistant$3e$__["useVoiceAssistant"])();
    const [screenShareTrack] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__t__as__useTracks$3e$__["useTracks"])([
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.ScreenShare
    ]);
    const cameraTrack = useLocalTrackRef(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Track"].Source.Camera);
    const isCameraEnabled = cameraTrack && !cameraTrack.publication.isMuted;
    const isScreenShareEnabled = screenShareTrack && !screenShareTrack.publication.isMuted;
    const hasSecondTile = isCameraEnabled || isScreenShareEnabled;
    const transition = {
        ...animationProps.transition,
        delay: chatOpen ? 0 : 0.15
    };
    const agentAnimate = {
        ...animationProps.animate,
        scale: chatOpen ? 1 : 3,
        transition
    };
    const avatarAnimate = {
        ...animationProps.animate,
        transition
    };
    const agentLayoutTransition = transition;
    const avatarLayoutTransition = transition;
    const isAvatar = agentVideoTrack !== undefined;
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "pointer-events-none fixed inset-x-0 top-8 bottom-32 z-50 md:top-12 md:bottom-40",
        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
            className: "relative mx-auto h-full max-w-2xl px-4 md:px-0",
            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])(classNames.grid),
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])([
                            'grid',
                            // 'bg-[hotpink]', // for debugging
                            !chatOpen && classNames.agentChatClosed,
                            chatOpen && hasSecondTile && classNames.agentChatOpenWithSecondTile,
                            chatOpen && !hasSecondTile && classNames.agentChatOpenWithoutSecondTile
                        ]),
                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$motion$40$12$2e$16$2e$0_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$motion$2f$dist$2f$es$2f$framer$2d$motion$2f$dist$2f$es$2f$components$2f$AnimatePresence$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["AnimatePresence"], {
                            mode: "popLayout",
                            children: [
                                !isAvatar && // audio-only agent
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(MotionAgentTile, {
                                    layoutId: "agent",
                                    ...animationProps,
                                    animate: agentAnimate,
                                    transition: agentLayoutTransition,
                                    state: agentState,
                                    audioTrack: agentAudioTrack,
                                    className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])(chatOpen ? 'h-[90px]' : 'h-auto w-full')
                                }, "agent", false, {
                                    fileName: "[project]/components/livekit/media-tiles.tsx",
                                    lineNumber: 141,
                                    columnNumber: 17
                                }, this),
                                isAvatar && // avatar agent
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(MotionAvatarTile, {
                                    layoutId: "avatar",
                                    ...animationProps,
                                    animate: avatarAnimate,
                                    transition: avatarLayoutTransition,
                                    videoTrack: agentVideoTrack,
                                    className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])(chatOpen ? 'h-[90px] [&>video]:h-[90px] [&>video]:w-auto' : 'h-auto w-full')
                                }, "avatar", false, {
                                    fileName: "[project]/components/livekit/media-tiles.tsx",
                                    lineNumber: 154,
                                    columnNumber: 17
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/components/livekit/media-tiles.tsx",
                            lineNumber: 138,
                            columnNumber: 13
                        }, this)
                    }, void 0, false, {
                        fileName: "[project]/components/livekit/media-tiles.tsx",
                        lineNumber: 129,
                        columnNumber: 11
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])([
                            'grid',
                            chatOpen && classNames.secondTileChatOpen,
                            !chatOpen && classNames.secondTileChatClosed
                        ]),
                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$motion$40$12$2e$16$2e$0_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$motion$2f$dist$2f$es$2f$framer$2d$motion$2f$dist$2f$es$2f$components$2f$AnimatePresence$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["AnimatePresence"], {
                            children: [
                                cameraTrack && isCameraEnabled && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(MotionVideoTile, {
                                    layout: "position",
                                    layoutId: "camera",
                                    ...animationProps,
                                    trackRef: cameraTrack,
                                    transition: {
                                        ...animationProps.transition,
                                        delay: chatOpen ? 0 : 0.15
                                    },
                                    className: "h-[90px]"
                                }, "camera", false, {
                                    fileName: "[project]/components/livekit/media-tiles.tsx",
                                    lineNumber: 179,
                                    columnNumber: 17
                                }, this),
                                isScreenShareEnabled && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(MotionVideoTile, {
                                    layout: "position",
                                    layoutId: "screen",
                                    ...animationProps,
                                    trackRef: screenShareTrack,
                                    transition: {
                                        ...animationProps.transition,
                                        delay: chatOpen ? 0 : 0.15
                                    },
                                    className: "h-[90px]"
                                }, "screen", false, {
                                    fileName: "[project]/components/livekit/media-tiles.tsx",
                                    lineNumber: 194,
                                    columnNumber: 17
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/components/livekit/media-tiles.tsx",
                            lineNumber: 177,
                            columnNumber: 13
                        }, this)
                    }, void 0, false, {
                        fileName: "[project]/components/livekit/media-tiles.tsx",
                        lineNumber: 169,
                        columnNumber: 11
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/components/livekit/media-tiles.tsx",
                lineNumber: 127,
                columnNumber: 9
            }, this)
        }, void 0, false, {
            fileName: "[project]/components/livekit/media-tiles.tsx",
            lineNumber: 126,
            columnNumber: 7
        }, this)
    }, void 0, false, {
        fileName: "[project]/components/livekit/media-tiles.tsx",
        lineNumber: 125,
        columnNumber: 5
    }, this);
}
}}),
"[project]/hooks/useChatAndTranscription.ts [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "default": (()=>useChatAndTranscription)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__w__as__useChat$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-ssr] (ecmascript) <export w as useChat>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__f__as__useRoomContext$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/contexts-CH4CA2DN.mjs [app-ssr] (ecmascript) <export f as useRoomContext>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__$5f$__as__useTranscriptions$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-ssr] (ecmascript) <export _ as useTranscriptions>");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/utils.ts [app-ssr] (ecmascript)");
;
;
;
function useChatAndTranscription() {
    const transcriptions = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__$5f$__as__useTranscriptions$3e$__["useTranscriptions"])();
    const chat = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__w__as__useChat$3e$__["useChat"])();
    const room = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__f__as__useRoomContext$3e$__["useRoomContext"])();
    const mergedTranscriptions = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useMemo"])(()=>{
        const merged = [
            ...transcriptions.map((transcription)=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["transcriptionToChatMessage"])(transcription, room)),
            ...chat.chatMessages
        ];
        return merged.sort((a, b)=>a.timestamp - b.timestamp);
    }, [
        transcriptions,
        chat.chatMessages,
        room
    ]);
    return {
        messages: mergedTranscriptions,
        send: chat.send
    };
}
}}),
"[project]/hooks/useDebug.ts [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "useDebugMode": (()=>useDebugMode)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-client@2.15.2_@types+dom-mediacapture-record@1.0.22/node_modules/livekit-client/dist/livekit-client.esm.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__f__as__useRoomContext$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/contexts-CH4CA2DN.mjs [app-ssr] (ecmascript) <export f as useRoomContext>");
;
;
;
const useDebugMode = ({ logLevel } = {})=>{
    const room = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__f__as__useRoomContext$3e$__["useRoomContext"])();
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["setLogLevel"])(logLevel ?? 'debug');
        // @ts-expect-error
        window.__lk_room = room;
        return ()=>{
            // @ts-expect-error
            window.__lk_room = undefined;
        };
    }, [
        room,
        logLevel
    ]);
};
}}),
"[project]/components/session-view.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "SessionView": (()=>SessionView)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$motion$40$12$2e$16$2e$0_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$motion$2f$dist$2f$es$2f$framer$2d$motion$2f$dist$2f$es$2f$components$2f$AnimatePresence$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/motion@12.16.0_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/motion/dist/es/framer-motion/dist/es/components/AnimatePresence/index.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$motion$40$12$2e$16$2e$0_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$motion$2f$dist$2f$es$2f$framer$2d$motion$2f$dist$2f$es$2f$render$2f$components$2f$motion$2f$proxy$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/motion@12.16.0_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/motion/dist/es/framer-motion/dist/es/render/components/motion/proxy.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__f__as__useRoomContext$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/contexts-CH4CA2DN.mjs [app-ssr] (ecmascript) <export f as useRoomContext>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__V__as__useVoiceAssistant$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/hooks-COF-7zxu.mjs [app-ssr] (ecmascript) <export V as useVoiceAssistant>");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$alert$2d$toast$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/alert-toast.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$agent$2d$control$2d$bar$2f$agent$2d$control$2d$bar$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/livekit/agent-control-bar/agent-control-bar.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$chat$2f$chat$2d$entry$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/livekit/chat/chat-entry.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$chat$2f$chat$2d$message$2d$view$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/livekit/chat/chat-message-view.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$media$2d$tiles$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/livekit/media-tiles.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useChatAndTranscription$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/hooks/useChatAndTranscription.ts [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useDebug$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/hooks/useDebug.ts [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/utils.ts [app-ssr] (ecmascript)");
'use client';
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
;
;
function isAgentAvailable(agentState) {
    return agentState == 'listening' || agentState == 'thinking' || agentState == 'speaking';
}
const SessionView = ({ appConfig, disabled, sessionStarted, ref })=>{
    const { state: agentState } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$hooks$2d$COF$2d$7zxu$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__V__as__useVoiceAssistant$3e$__["useVoiceAssistant"])();
    const [chatOpen, setChatOpen] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(false);
    const { messages, send } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useChatAndTranscription$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["default"])();
    const room = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__f__as__useRoomContext$3e$__["useRoomContext"])();
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useDebug$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useDebugMode"])();
    async function handleSendMessage(message) {
        await send(message);
    }
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        if (sessionStarted) {
            const timeout = setTimeout(()=>{
                if (!isAgentAvailable(agentState)) {
                    const reason = agentState === 'connecting' ? 'Agent did not join the room. ' : 'Agent connected but did not complete initializing. ';
                    (0, __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$alert$2d$toast$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["toastAlert"])({
                        title: 'Session ended',
                        description: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                            className: "w-full",
                            children: [
                                reason,
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("a", {
                                    target: "_blank",
                                    rel: "noopener noreferrer",
                                    href: "https://docs.livekit.io/agents/start/voice-ai/",
                                    className: "whitespace-nowrap underline",
                                    children: "See quickstart guide"
                                }, void 0, false, {
                                    fileName: "[project]/components/session-view.tsx",
                                    lineNumber: 62,
                                    columnNumber: 17
                                }, this),
                                "."
                            ]
                        }, void 0, true, {
                            fileName: "[project]/components/session-view.tsx",
                            lineNumber: 60,
                            columnNumber: 15
                        }, this)
                    });
                    room.disconnect();
                }
            }, 10_000);
            return ()=>clearTimeout(timeout);
        }
    }, [
        agentState,
        sessionStarted,
        room
    ]);
    const { supportsChatInput, supportsVideoInput, supportsScreenShare } = appConfig;
    const capabilities = {
        supportsChatInput,
        supportsVideoInput,
        supportsScreenShare
    };
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("main", {
        ref: ref,
        inert: disabled,
        className: // prevent page scrollbar
        // when !chatOpen due to 'translate-y-20'
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])(!chatOpen && 'max-h-svh overflow-hidden'),
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$chat$2f$chat$2d$message$2d$view$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["ChatMessageView"], {
                className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])('mx-auto min-h-svh w-full max-w-2xl px-3 pt-32 pb-40 transition-[opacity,translate] duration-300 ease-out md:px-0 md:pt-36 md:pb-48', chatOpen ? 'translate-y-0 opacity-100 delay-200' : 'translate-y-20 opacity-0'),
                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    className: "space-y-3 whitespace-pre-wrap",
                    children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$motion$40$12$2e$16$2e$0_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$motion$2f$dist$2f$es$2f$framer$2d$motion$2f$dist$2f$es$2f$components$2f$AnimatePresence$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["AnimatePresence"], {
                        children: messages.map((message)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$motion$40$12$2e$16$2e$0_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$motion$2f$dist$2f$es$2f$framer$2d$motion$2f$dist$2f$es$2f$render$2f$components$2f$motion$2f$proxy$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["motion"].div, {
                                initial: {
                                    opacity: 0,
                                    height: 0
                                },
                                animate: {
                                    opacity: 1,
                                    height: 'auto'
                                },
                                exit: {
                                    opacity: 1,
                                    height: 'auto',
                                    translateY: 0.001
                                },
                                transition: {
                                    duration: 0.5,
                                    ease: 'easeOut'
                                },
                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$chat$2f$chat$2d$entry$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["ChatEntry"], {
                                    hideName: true,
                                    entry: message
                                }, message.id, false, {
                                    fileName: "[project]/components/session-view.tsx",
                                    lineNumber: 115,
                                    columnNumber: 17
                                }, this)
                            }, message.id, false, {
                                fileName: "[project]/components/session-view.tsx",
                                lineNumber: 108,
                                columnNumber: 15
                            }, this))
                    }, void 0, false, {
                        fileName: "[project]/components/session-view.tsx",
                        lineNumber: 106,
                        columnNumber: 11
                    }, this)
                }, void 0, false, {
                    fileName: "[project]/components/session-view.tsx",
                    lineNumber: 105,
                    columnNumber: 9
                }, this)
            }, void 0, false, {
                fileName: "[project]/components/session-view.tsx",
                lineNumber: 99,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "bg-background mp-12 fixed top-0 right-0 left-0 h-32 md:h-36",
                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    className: "from-background absolute bottom-0 left-0 h-12 w-full translate-y-full bg-gradient-to-b to-transparent"
                }, void 0, false, {
                    fileName: "[project]/components/session-view.tsx",
                    lineNumber: 124,
                    columnNumber: 9
                }, this)
            }, void 0, false, {
                fileName: "[project]/components/session-view.tsx",
                lineNumber: 122,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$media$2d$tiles$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["MediaTiles"], {
                chatOpen: chatOpen
            }, void 0, false, {
                fileName: "[project]/components/session-view.tsx",
                lineNumber: 127,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "bg-background fixed right-0 bottom-0 left-0 z-50 px-3 pt-2 pb-3 md:px-12 md:pb-12",
                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$motion$40$12$2e$16$2e$0_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$motion$2f$dist$2f$es$2f$framer$2d$motion$2f$dist$2f$es$2f$render$2f$components$2f$motion$2f$proxy$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["motion"].div, {
                    initial: {
                        opacity: 0,
                        translateY: '100%'
                    },
                    animate: {
                        opacity: sessionStarted ? 1 : 0,
                        translateY: sessionStarted ? '0%' : '100%'
                    },
                    transition: {
                        duration: 0.3,
                        delay: sessionStarted ? 0.5 : 0,
                        ease: 'easeOut'
                    },
                    children: [
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "relative z-10 mx-auto w-full max-w-2xl",
                            children: [
                                appConfig.isPreConnectBufferEnabled && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$motion$40$12$2e$16$2e$0_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$motion$2f$dist$2f$es$2f$framer$2d$motion$2f$dist$2f$es$2f$render$2f$components$2f$motion$2f$proxy$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["motion"].div, {
                                    initial: {
                                        opacity: 0
                                    },
                                    animate: {
                                        opacity: sessionStarted && messages.length === 0 ? 1 : 0,
                                        transition: {
                                            ease: 'easeIn',
                                            delay: messages.length > 0 ? 0 : 0.8,
                                            duration: messages.length > 0 ? 0.2 : 0.5
                                        }
                                    },
                                    "aria-hidden": messages.length > 0,
                                    className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])('absolute inset-x-0 -top-12 text-center', sessionStarted && messages.length === 0 && 'pointer-events-none'),
                                    children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "animate-text-shimmer inline-block !bg-clip-text text-sm font-semibold text-transparent",
                                        children: "Agent is listening, ask it a question"
                                    }, void 0, false, {
                                        fileName: "[project]/components/session-view.tsx",
                                        lineNumber: 157,
                                        columnNumber: 17
                                    }, this)
                                }, void 0, false, {
                                    fileName: "[project]/components/session-view.tsx",
                                    lineNumber: 141,
                                    columnNumber: 15
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$livekit$2f$agent$2d$control$2d$bar$2f$agent$2d$control$2d$bar$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["AgentControlBar"], {
                                    capabilities: capabilities,
                                    onChatOpenChange: setChatOpen,
                                    onSendMessage: handleSendMessage
                                }, void 0, false, {
                                    fileName: "[project]/components/session-view.tsx",
                                    lineNumber: 163,
                                    columnNumber: 13
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/components/session-view.tsx",
                            lineNumber: 139,
                            columnNumber: 11
                        }, this),
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "from-background border-background absolute top-0 left-0 h-12 w-full -translate-y-full bg-gradient-to-t to-transparent"
                        }, void 0, false, {
                            fileName: "[project]/components/session-view.tsx",
                            lineNumber: 170,
                            columnNumber: 11
                        }, this)
                    ]
                }, "control-bar", true, {
                    fileName: "[project]/components/session-view.tsx",
                    lineNumber: 130,
                    columnNumber: 9
                }, this)
            }, void 0, false, {
                fileName: "[project]/components/session-view.tsx",
                lineNumber: 129,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/components/session-view.tsx",
        lineNumber: 90,
        columnNumber: 5
    }, this);
};
}}),
"[project]/components/ui/sonner.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "Toaster": (()=>Toaster)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$2d$themes$40$0$2e$4$2e$6_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2d$themes$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next-themes@0.4.6_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next-themes/dist/index.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$sonner$40$2$2e$0$2e$6_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$sonner$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/sonner@2.0.6_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/sonner/dist/index.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$Warning$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@phosphor-icons+react@2.1.10_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/@phosphor-icons/react/dist/ssr/Warning.es.js [app-ssr] (ecmascript)");
'use client';
;
;
;
;
const Toaster = ({ ...props })=>{
    const { theme = 'system' } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$2d$themes$40$0$2e$4$2e$6_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2d$themes$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useTheme"])();
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$sonner$40$2$2e$0$2e$6_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$sonner$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Toaster"], {
        theme: theme,
        className: "toaster group",
        position: "top-center",
        icons: {
            warning: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$phosphor$2d$icons$2b$react$40$2$2e$1$2e$10_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f40$phosphor$2d$icons$2f$react$2f$dist$2f$ssr$2f$Warning$2e$es$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["WarningIcon"], {
                weight: "bold"
            }, void 0, false, {
                fileName: "[project]/components/ui/sonner.tsx",
                lineNumber: 16,
                columnNumber: 18
            }, void 0)
        },
        style: {
            '--normal-bg': 'var(--popover)',
            '--normal-text': 'var(--popover-foreground)',
            '--normal-border': 'var(--border)'
        },
        ...props
    }, void 0, false, {
        fileName: "[project]/components/ui/sonner.tsx",
        lineNumber: 11,
        columnNumber: 5
    }, this);
};
;
}}),
"[project]/components/welcome.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "Welcome": (()=>Welcome)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$button$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/ui/button.tsx [app-ssr] (ecmascript)");
;
;
const Welcome = ({ disabled, startButtonText, onStartCall, ref })=>{
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        ref: ref,
        inert: disabled,
        className: "fixed inset-0 z-10 mx-auto flex h-svh flex-col items-center justify-center text-center",
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                width: "64",
                height: "64",
                viewBox: "0 0 64 64",
                fill: "none",
                xmlns: "http://www.w3.org/2000/svg",
                className: "text-fg0 mb-4 size-16",
                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                    d: "M15 24V40C15 40.7957 14.6839 41.5587 14.1213 42.1213C13.5587 42.6839 12.7956 43 12 43C11.2044 43 10.4413 42.6839 9.87868 42.1213C9.31607 41.5587 9 40.7957 9 40V24C9 23.2044 9.31607 22.4413 9.87868 21.8787C10.4413 21.3161 11.2044 21 12 21C12.7956 21 13.5587 21.3161 14.1213 21.8787C14.6839 22.4413 15 23.2044 15 24ZM22 5C21.2044 5 20.4413 5.31607 19.8787 5.87868C19.3161 6.44129 19 7.20435 19 8V56C19 56.7957 19.3161 57.5587 19.8787 58.1213C20.4413 58.6839 21.2044 59 22 59C22.7956 59 23.5587 58.6839 24.1213 58.1213C24.6839 57.5587 25 56.7957 25 56V8C25 7.20435 24.6839 6.44129 24.1213 5.87868C23.5587 5.31607 22.7956 5 22 5ZM32 13C31.2044 13 30.4413 13.3161 29.8787 13.8787C29.3161 14.4413 29 15.2044 29 16V48C29 48.7957 29.3161 49.5587 29.8787 50.1213C30.4413 50.6839 31.2044 51 32 51C32.7956 51 33.5587 50.6839 34.1213 50.1213C34.6839 49.5587 35 48.7957 35 48V16C35 15.2044 34.6839 14.4413 34.1213 13.8787C33.5587 13.3161 32.7956 13 32 13ZM42 21C41.2043 21 40.4413 21.3161 39.8787 21.8787C39.3161 22.4413 39 23.2044 39 24V40C39 40.7957 39.3161 41.5587 39.8787 42.1213C40.4413 42.6839 41.2043 43 42 43C42.7957 43 43.5587 42.6839 44.1213 42.1213C44.6839 41.5587 45 40.7957 45 40V24C45 23.2044 44.6839 22.4413 44.1213 21.8787C43.5587 21.3161 42.7957 21 42 21ZM52 17C51.2043 17 50.4413 17.3161 49.8787 17.8787C49.3161 18.4413 49 19.2044 49 20V44C49 44.7957 49.3161 45.5587 49.8787 46.1213C50.4413 46.6839 51.2043 47 52 47C52.7957 47 53.5587 46.6839 54.1213 46.1213C54.6839 45.5587 55 44.7957 55 44V20C55 19.2044 54.6839 18.4413 54.1213 17.8787C53.5587 17.3161 52.7957 17 52 17Z",
                    fill: "currentColor"
                }, void 0, false, {
                    fileName: "[project]/components/welcome.tsx",
                    lineNumber: 29,
                    columnNumber: 9
                }, this)
            }, void 0, false, {
                fileName: "[project]/components/welcome.tsx",
                lineNumber: 21,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                className: "text-fg1 max-w-prose pt-1 leading-6 font-medium",
                children: "Chat live with your voice AI agent"
            }, void 0, false, {
                fileName: "[project]/components/welcome.tsx",
                lineNumber: 35,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$button$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Button"], {
                variant: "primary",
                size: "lg",
                onClick: onStartCall,
                className: "mt-6 w-64 font-mono",
                children: startButtonText
            }, void 0, false, {
                fileName: "[project]/components/welcome.tsx",
                lineNumber: 38,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                className: "text-fg1 m fixed bottom-5 left-1/2 w-full max-w-prose -translate-x-1/2 pt-1 text-xs leading-5 font-normal text-pretty md:text-sm",
                children: [
                    "Need help getting set up? Check out the",
                    ' ',
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("a", {
                        target: "_blank",
                        rel: "noopener noreferrer",
                        href: "https://docs.livekit.io/agents/start/voice-ai/",
                        className: "underline",
                        children: "Voice AI quickstart"
                    }, void 0, false, {
                        fileName: "[project]/components/welcome.tsx",
                        lineNumber: 43,
                        columnNumber: 9
                    }, this),
                    "."
                ]
            }, void 0, true, {
                fileName: "[project]/components/welcome.tsx",
                lineNumber: 41,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/components/welcome.tsx",
        lineNumber: 16,
        columnNumber: 5
    }, this);
};
}}),
"[project]/hooks/useConnectionDetails.ts [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "default": (()=>useConnectionDetails)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
;
function useConnectionDetails() {
    // Generate room connection details, including:
    //   - A random Room name
    //   - A random Participant name
    //   - An Access Token to permit the participant to join the room
    //   - The URL of the LiveKit server to connect to
    //
    // In real-world application, you would likely allow the user to specify their
    // own participant name, and possibly to choose from existing rooms to join.
    const [connectionDetails, setConnectionDetails] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(null);
    const fetchConnectionDetails = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useCallback"])(()=>{
        setConnectionDetails(null);
        const url = new URL(process.env.NEXT_PUBLIC_CONN_DETAILS_ENDPOINT ?? '/api/connection-details', window.location.origin);
        fetch(url.toString()).then((res)=>res.json()).then((data)=>{
            setConnectionDetails(data);
        }).catch((error)=>{
            console.error('Error fetching connection details:', error);
        });
    }, []);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        fetchConnectionDetails();
    }, [
        fetchConnectionDetails
    ]);
    return {
        connectionDetails,
        refreshConnectionDetails: fetchConnectionDetails
    };
}
}}),
"[project]/components/app.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "App": (()=>App)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/next@15.3.5_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/livekit-client@2.15.2_@types+dom-mediacapture-record@1.0.22/node_modules/livekit-client/dist/livekit-client.esm.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$motion$40$12$2e$16$2e$0_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$motion$2f$dist$2f$es$2f$framer$2d$motion$2f$dist$2f$es$2f$render$2f$components$2f$motion$2f$proxy$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/motion@12.16.0_react-dom@19.1.0_react@19.1.0__react@19.1.0/node_modules/motion/dist/es/framer-motion/dist/es/render/components/motion/proxy.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__R__as__RoomAudioRenderer$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/components-k0KtCs0w.mjs [app-ssr] (ecmascript) <export R as RoomAudioRenderer>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__R__as__RoomContext$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/contexts-CH4CA2DN.mjs [app-ssr] (ecmascript) <export R as RoomContext>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__q__as__StartAudio$3e$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@livekit+components-react@2.9.13_@livekit+krisp-noise-filter@0.2.16_livekit-client@2.15.2_@ty_rg3q5pgpndloylo6swz3k6u5xm/node_modules/@livekit/components-react/dist/components-k0KtCs0w.mjs [app-ssr] (ecmascript) <export q as StartAudio>");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$alert$2d$toast$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/alert-toast.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$session$2d$view$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/session-view.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$sonner$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/ui/sonner.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$welcome$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/components/welcome.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useConnectionDetails$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/hooks/useConnectionDetails.ts [app-ssr] (ecmascript)");
'use client';
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
const MotionWelcome = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$motion$40$12$2e$16$2e$0_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$motion$2f$dist$2f$es$2f$framer$2d$motion$2f$dist$2f$es$2f$render$2f$components$2f$motion$2f$proxy$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["motion"].create(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$welcome$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Welcome"]);
const MotionSessionView = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$motion$40$12$2e$16$2e$0_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$motion$2f$dist$2f$es$2f$framer$2d$motion$2f$dist$2f$es$2f$render$2f$components$2f$motion$2f$proxy$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["motion"].create(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$session$2d$view$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["SessionView"]);
function App({ appConfig }) {
    const room = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useMemo"])(()=>new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Room"](), []);
    const [sessionStarted, setSessionStarted] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(false);
    const { connectionDetails, refreshConnectionDetails } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useConnectionDetails$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["default"])();
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        const onDisconnected = ()=>{
            setSessionStarted(false);
            refreshConnectionDetails();
        };
        const onMediaDevicesError = (error)=>{
            (0, __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$alert$2d$toast$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["toastAlert"])({
                title: 'Encountered an error with your media devices',
                description: `${error.name}: ${error.message}`
            });
        };
        room.on(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["RoomEvent"].MediaDevicesError, onMediaDevicesError);
        room.on(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["RoomEvent"].Disconnected, onDisconnected);
        return ()=>{
            room.off(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["RoomEvent"].Disconnected, onDisconnected);
            room.off(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$livekit$2d$client$40$2$2e$15$2e$2_$40$types$2b$dom$2d$mediacapture$2d$record$40$1$2e$0$2e$22$2f$node_modules$2f$livekit$2d$client$2f$dist$2f$livekit$2d$client$2e$esm$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["RoomEvent"].MediaDevicesError, onMediaDevicesError);
        };
    }, [
        room,
        refreshConnectionDetails
    ]);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        if (sessionStarted && room.state === 'disconnected' && connectionDetails) {
            Promise.all([
                room.localParticipant.setMicrophoneEnabled(true, undefined, {
                    preConnectBuffer: appConfig.isPreConnectBufferEnabled
                }),
                room.connect(connectionDetails.serverUrl, connectionDetails.participantToken)
            ]).catch((error)=>{
                (0, __TURBOPACK__imported__module__$5b$project$5d2f$components$2f$alert$2d$toast$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["toastAlert"])({
                    title: 'There was an error connecting to the agent',
                    description: `${error.name}: ${error.message}`
                });
            });
        }
        return ()=>{
            room.disconnect();
        };
    }, [
        room,
        sessionStarted,
        connectionDetails,
        appConfig.isPreConnectBufferEnabled
    ]);
    const { startButtonText } = appConfig;
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Fragment"], {
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(MotionWelcome, {
                startButtonText: startButtonText,
                onStartCall: ()=>setSessionStarted(true),
                disabled: sessionStarted,
                initial: {
                    opacity: 0
                },
                animate: {
                    opacity: sessionStarted ? 0 : 1
                },
                transition: {
                    duration: 0.5,
                    ease: 'linear',
                    delay: sessionStarted ? 0 : 0.5
                }
            }, "welcome", false, {
                fileName: "[project]/components/app.tsx",
                lineNumber: 68,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$contexts$2d$CH4CA2DN$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__R__as__RoomContext$3e$__["RoomContext"].Provider, {
                value: room,
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__R__as__RoomAudioRenderer$3e$__["RoomAudioRenderer"], {}, void 0, false, {
                        fileName: "[project]/components/app.tsx",
                        lineNumber: 79,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$livekit$2b$components$2d$react$40$2$2e$9$2e$13_$40$livekit$2b$krisp$2d$noise$2d$filter$40$0$2e$2$2e$16_livekit$2d$client$40$2$2e$15$2e$2_$40$ty_rg3q5pgpndloylo6swz3k6u5xm$2f$node_modules$2f40$livekit$2f$components$2d$react$2f$dist$2f$components$2d$k0KtCs0w$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__q__as__StartAudio$3e$__["StartAudio"], {
                        label: "Start Audio"
                    }, void 0, false, {
                        fileName: "[project]/components/app.tsx",
                        lineNumber: 80,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(MotionSessionView, {
                        appConfig: appConfig,
                        disabled: !sessionStarted,
                        sessionStarted: sessionStarted,
                        initial: {
                            opacity: 0
                        },
                        animate: {
                            opacity: sessionStarted ? 1 : 0
                        },
                        transition: {
                            duration: 0.5,
                            ease: 'linear',
                            delay: sessionStarted ? 0.5 : 0
                        }
                    }, "session-view", false, {
                        fileName: "[project]/components/app.tsx",
                        lineNumber: 82,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/components/app.tsx",
                lineNumber: 78,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f$next$40$15$2e$3$2e$5_react$2d$dom$40$19$2e$1$2e$0_react$40$19$2e$1$2e$0_$5f$react$40$19$2e$1$2e$0$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$components$2f$ui$2f$sonner$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Toaster"], {}, void 0, false, {
                fileName: "[project]/components/app.tsx",
                lineNumber: 97,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true);
}
}}),

};

//# sourceMappingURL=_78d265de._.js.map