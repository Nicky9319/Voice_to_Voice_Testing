module.exports = {

"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/assert.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * Assert that condition is truthy or throw error (with message)
 */ __turbopack_context__.s({
    "assert": (()=>assert),
    "assertFloat32": (()=>assertFloat32),
    "assertInt32": (()=>assertInt32),
    "assertUInt32": (()=>assertUInt32)
});
function assert(condition, msg) {
    // eslint-disable-next-line @typescript-eslint/strict-boolean-expressions -- we want the implicit conversion to boolean
    if (!condition) {
        throw new Error(msg);
    }
}
const FLOAT32_MAX = 3.4028234663852886e38, FLOAT32_MIN = -3.4028234663852886e38, UINT32_MAX = 0xffffffff, INT32_MAX = 0x7fffffff, INT32_MIN = -0x80000000;
function assertInt32(arg) {
    if (typeof arg !== "number") throw new Error("invalid int 32: " + typeof arg);
    if (!Number.isInteger(arg) || arg > INT32_MAX || arg < INT32_MIN) throw new Error("invalid int 32: " + arg); // eslint-disable-line @typescript-eslint/restrict-plus-operands -- we want the implicit conversion to string
}
function assertUInt32(arg) {
    if (typeof arg !== "number") throw new Error("invalid uint 32: " + typeof arg);
    if (!Number.isInteger(arg) || arg > UINT32_MAX || arg < 0) throw new Error("invalid uint 32: " + arg); // eslint-disable-line @typescript-eslint/restrict-plus-operands -- we want the implicit conversion to string
}
function assertFloat32(arg) {
    if (typeof arg !== "number") throw new Error("invalid float 32: " + typeof arg);
    if (!Number.isFinite(arg)) return;
    if (arg > FLOAT32_MAX || arg < FLOAT32_MIN) throw new Error("invalid float 32: " + arg); // eslint-disable-line @typescript-eslint/restrict-plus-operands -- we want the implicit conversion to string
}
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/enum.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "getEnumType": (()=>getEnumType),
    "makeEnum": (()=>makeEnum),
    "makeEnumType": (()=>makeEnumType),
    "setEnumType": (()=>setEnumType)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/assert.js [app-route] (ecmascript)");
;
const enumTypeSymbol = Symbol("@bufbuild/protobuf/enum-type");
function getEnumType(enumObject) {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access,@typescript-eslint/no-unsafe-assignment,@typescript-eslint/no-explicit-any
    const t = enumObject[enumTypeSymbol];
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(t, "missing enum type on enum object");
    return t; // eslint-disable-line @typescript-eslint/no-unsafe-return
}
function setEnumType(enumObject, typeName, values, opt) {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any
    enumObject[enumTypeSymbol] = makeEnumType(typeName, values.map((v)=>({
            no: v.no,
            name: v.name,
            localName: enumObject[v.no]
        })), opt);
}
function makeEnumType(typeName, values, // eslint-disable-next-line @typescript-eslint/no-unused-vars
_opt) {
    const names = Object.create(null);
    const numbers = Object.create(null);
    const normalValues = [];
    for (const value of values){
        // We do not surface options at this time
        // const value: EnumValueInfo = {...v, options: v.options ?? emptyReadonlyObject};
        const n = normalizeEnumValue(value);
        normalValues.push(n);
        names[value.name] = n;
        numbers[value.no] = n;
    }
    return {
        typeName,
        values: normalValues,
        // We do not surface options at this time
        // options: opt?.options ?? Object.create(null),
        findName (name) {
            return names[name];
        },
        findNumber (no) {
            return numbers[no];
        }
    };
}
function makeEnum(typeName, values, opt) {
    const enumObject = {};
    for (const value of values){
        const n = normalizeEnumValue(value);
        enumObject[n.localName] = n.no;
        enumObject[n.no] = n.localName;
    }
    setEnumType(enumObject, typeName, values, opt);
    return enumObject;
}
function normalizeEnumValue(value) {
    if ("localName" in value) {
        return value;
    }
    return Object.assign(Object.assign({}, value), {
        localName: value.name
    });
}
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/message.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * Message is the base class of every message, generated, or created at
 * runtime.
 *
 * It is _not_ safe to extend this class. If you want to create a message at
 * run time, use proto3.makeMessageType().
 */ __turbopack_context__.s({
    "Message": (()=>Message)
});
class Message {
    /**
     * Compare with a message of the same type.
     * Note that this function disregards extensions and unknown fields.
     */ equals(other) {
        return this.getType().runtime.util.equals(this.getType(), this, other);
    }
    /**
     * Create a deep copy.
     */ clone() {
        return this.getType().runtime.util.clone(this);
    }
    /**
     * Parse from binary data, merging fields.
     *
     * Repeated fields are appended. Map entries are added, overwriting
     * existing keys.
     *
     * If a message field is already present, it will be merged with the
     * new data.
     */ fromBinary(bytes, options) {
        const type = this.getType(), format = type.runtime.bin, opt = format.makeReadOptions(options);
        format.readMessage(this, opt.readerFactory(bytes), bytes.byteLength, opt);
        return this;
    }
    /**
     * Parse a message from a JSON value.
     */ fromJson(jsonValue, options) {
        const type = this.getType(), format = type.runtime.json, opt = format.makeReadOptions(options);
        format.readMessage(type, jsonValue, opt, this);
        return this;
    }
    /**
     * Parse a message from a JSON string.
     */ fromJsonString(jsonString, options) {
        let json;
        try {
            json = JSON.parse(jsonString);
        } catch (e) {
            throw new Error(`cannot decode ${this.getType().typeName} from JSON: ${e instanceof Error ? e.message : String(e)}`);
        }
        return this.fromJson(json, options);
    }
    /**
     * Serialize the message to binary data.
     */ toBinary(options) {
        const type = this.getType(), bin = type.runtime.bin, opt = bin.makeWriteOptions(options), writer = opt.writerFactory();
        bin.writeMessage(this, writer, opt);
        return writer.finish();
    }
    /**
     * Serialize the message to a JSON value, a JavaScript value that can be
     * passed to JSON.stringify().
     */ toJson(options) {
        const type = this.getType(), json = type.runtime.json, opt = json.makeWriteOptions(options);
        return json.writeMessage(this, opt);
    }
    /**
     * Serialize the message to a JSON string.
     */ toJsonString(options) {
        var _a;
        const value = this.toJson(options);
        return JSON.stringify(value, null, (_a = options === null || options === void 0 ? void 0 : options.prettySpaces) !== null && _a !== void 0 ? _a : 0);
    }
    /**
     * Override for serialization behavior. This will be invoked when calling
     * JSON.stringify on this message (i.e. JSON.stringify(msg)).
     *
     * Note that this will not serialize google.protobuf.Any with a packed
     * message because the protobuf JSON format specifies that it needs to be
     * unpacked, and this is only possible with a type registry to look up the
     * message type.  As a result, attempting to serialize a message with this
     * type will throw an Error.
     *
     * This method is protected because you should not need to invoke it
     * directly -- instead use JSON.stringify or toJsonString for
     * stringified JSON.  Alternatively, if actual JSON is desired, you should
     * use toJson.
     */ toJSON() {
        return this.toJson({
            emitDefaultValues: true
        });
    }
    /**
     * Retrieve the MessageType of this message - a singleton that represents
     * the protobuf message declaration and provides metadata for reflection-
     * based operations.
     */ getType() {
        // Any class that extends Message _must_ provide a complete static
        // implementation of MessageType.
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access,@typescript-eslint/no-unsafe-return
        return Object.getPrototypeOf(this).constructor;
    }
}
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/message-type.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "makeMessageType": (()=>makeMessageType)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/message.js [app-route] (ecmascript)");
;
function makeMessageType(runtime, typeName, fields, opt) {
    var _a;
    const localName = (_a = opt === null || opt === void 0 ? void 0 : opt.localName) !== null && _a !== void 0 ? _a : typeName.substring(typeName.lastIndexOf(".") + 1);
    const type = {
        [localName]: function(data) {
            runtime.util.initFields(this);
            runtime.util.initPartial(data, this);
        }
    }[localName];
    Object.setPrototypeOf(type.prototype, new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["Message"]());
    Object.assign(type, {
        runtime,
        typeName,
        fields: runtime.util.newFieldList(fields),
        fromBinary (bytes, options) {
            return new type().fromBinary(bytes, options);
        },
        fromJson (jsonValue, options) {
            return new type().fromJson(jsonValue, options);
        },
        fromJsonString (jsonString, options) {
            return new type().fromJsonString(jsonString, options);
        },
        equals (a, b) {
            return runtime.util.equals(type, a, b);
        }
    });
    return type;
}
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/google/varint.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2008 Google Inc.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Code generated by the Protocol Buffer compiler is owned by the owner
// of the input file used when generating it.  This code is not
// standalone and requires a support library to be linked with it.  This
// support library is itself covered by the above license.
/* eslint-disable prefer-const,@typescript-eslint/restrict-plus-operands */ /**
 * Read a 64 bit varint as two JS numbers.
 *
 * Returns tuple:
 * [0]: low bits
 * [1]: high bits
 *
 * Copyright 2008 Google Inc.  All rights reserved.
 *
 * See https://github.com/protocolbuffers/protobuf/blob/8a71927d74a4ce34efe2d8769fda198f52d20d12/js/experimental/runtime/kernel/buffer_decoder.js#L175
 */ __turbopack_context__.s({
    "int64FromString": (()=>int64FromString),
    "int64ToString": (()=>int64ToString),
    "uInt64ToString": (()=>uInt64ToString),
    "varint32read": (()=>varint32read),
    "varint32write": (()=>varint32write),
    "varint64read": (()=>varint64read),
    "varint64write": (()=>varint64write)
});
function varint64read() {
    let lowBits = 0;
    let highBits = 0;
    for(let shift = 0; shift < 28; shift += 7){
        let b = this.buf[this.pos++];
        lowBits |= (b & 0x7f) << shift;
        if ((b & 0x80) == 0) {
            this.assertBounds();
            return [
                lowBits,
                highBits
            ];
        }
    }
    let middleByte = this.buf[this.pos++];
    // last four bits of the first 32 bit number
    lowBits |= (middleByte & 0x0f) << 28;
    // 3 upper bits are part of the next 32 bit number
    highBits = (middleByte & 0x70) >> 4;
    if ((middleByte & 0x80) == 0) {
        this.assertBounds();
        return [
            lowBits,
            highBits
        ];
    }
    for(let shift = 3; shift <= 31; shift += 7){
        let b = this.buf[this.pos++];
        highBits |= (b & 0x7f) << shift;
        if ((b & 0x80) == 0) {
            this.assertBounds();
            return [
                lowBits,
                highBits
            ];
        }
    }
    throw new Error("invalid varint");
}
function varint64write(lo, hi, bytes) {
    for(let i = 0; i < 28; i = i + 7){
        const shift = lo >>> i;
        const hasNext = !(shift >>> 7 == 0 && hi == 0);
        const byte = (hasNext ? shift | 0x80 : shift) & 0xff;
        bytes.push(byte);
        if (!hasNext) {
            return;
        }
    }
    const splitBits = lo >>> 28 & 0x0f | (hi & 0x07) << 4;
    const hasMoreBits = !(hi >> 3 == 0);
    bytes.push((hasMoreBits ? splitBits | 0x80 : splitBits) & 0xff);
    if (!hasMoreBits) {
        return;
    }
    for(let i = 3; i < 31; i = i + 7){
        const shift = hi >>> i;
        const hasNext = !(shift >>> 7 == 0);
        const byte = (hasNext ? shift | 0x80 : shift) & 0xff;
        bytes.push(byte);
        if (!hasNext) {
            return;
        }
    }
    bytes.push(hi >>> 31 & 0x01);
}
// constants for binary math
const TWO_PWR_32_DBL = 0x100000000;
function int64FromString(dec) {
    // Check for minus sign.
    const minus = dec[0] === "-";
    if (minus) {
        dec = dec.slice(1);
    }
    // Work 6 decimal digits at a time, acting like we're converting base 1e6
    // digits to binary. This is safe to do with floating point math because
    // Number.isSafeInteger(ALL_32_BITS * 1e6) == true.
    const base = 1e6;
    let lowBits = 0;
    let highBits = 0;
    function add1e6digit(begin, end) {
        // Note: Number('') is 0.
        const digit1e6 = Number(dec.slice(begin, end));
        highBits *= base;
        lowBits = lowBits * base + digit1e6;
        // Carry bits from lowBits to
        if (lowBits >= TWO_PWR_32_DBL) {
            highBits = highBits + (lowBits / TWO_PWR_32_DBL | 0);
            lowBits = lowBits % TWO_PWR_32_DBL;
        }
    }
    add1e6digit(-24, -18);
    add1e6digit(-18, -12);
    add1e6digit(-12, -6);
    add1e6digit(-6);
    return minus ? negate(lowBits, highBits) : newBits(lowBits, highBits);
}
function int64ToString(lo, hi) {
    let bits = newBits(lo, hi);
    // If we're treating the input as a signed value and the high bit is set, do
    // a manual two's complement conversion before the decimal conversion.
    const negative = bits.hi & 0x80000000;
    if (negative) {
        bits = negate(bits.lo, bits.hi);
    }
    const result = uInt64ToString(bits.lo, bits.hi);
    return negative ? "-" + result : result;
}
function uInt64ToString(lo, hi) {
    ({ lo, hi } = toUnsigned(lo, hi));
    // Skip the expensive conversion if the number is small enough to use the
    // built-in conversions.
    // Number.MAX_SAFE_INTEGER = 0x001FFFFF FFFFFFFF, thus any number with
    // highBits <= 0x1FFFFF can be safely expressed with a double and retain
    // integer precision.
    // Proven by: Number.isSafeInteger(0x1FFFFF * 2**32 + 0xFFFFFFFF) == true.
    if (hi <= 0x1FFFFF) {
        return String(TWO_PWR_32_DBL * hi + lo);
    }
    // What this code is doing is essentially converting the input number from
    // base-2 to base-1e7, which allows us to represent the 64-bit range with
    // only 3 (very large) digits. Those digits are then trivial to convert to
    // a base-10 string.
    // The magic numbers used here are -
    // 2^24 = 16777216 = (1,6777216) in base-1e7.
    // 2^48 = 281474976710656 = (2,8147497,6710656) in base-1e7.
    // Split 32:32 representation into 16:24:24 representation so our
    // intermediate digits don't overflow.
    const low = lo & 0xFFFFFF;
    const mid = (lo >>> 24 | hi << 8) & 0xFFFFFF;
    const high = hi >> 16 & 0xFFFF;
    // Assemble our three base-1e7 digits, ignoring carries. The maximum
    // value in a digit at this step is representable as a 48-bit integer, which
    // can be stored in a 64-bit floating point number.
    let digitA = low + mid * 6777216 + high * 6710656;
    let digitB = mid + high * 8147497;
    let digitC = high * 2;
    // Apply carries from A to B and from B to C.
    const base = 10000000;
    if (digitA >= base) {
        digitB += Math.floor(digitA / base);
        digitA %= base;
    }
    if (digitB >= base) {
        digitC += Math.floor(digitB / base);
        digitB %= base;
    }
    // If digitC is 0, then we should have returned in the trivial code path
    // at the top for non-safe integers. Given this, we can assume both digitB
    // and digitA need leading zeros.
    return digitC.toString() + decimalFrom1e7WithLeadingZeros(digitB) + decimalFrom1e7WithLeadingZeros(digitA);
}
function toUnsigned(lo, hi) {
    return {
        lo: lo >>> 0,
        hi: hi >>> 0
    };
}
function newBits(lo, hi) {
    return {
        lo: lo | 0,
        hi: hi | 0
    };
}
/**
 * Returns two's compliment negation of input.
 * @see https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Bitwise_Operators#Signed_32-bit_integers
 */ function negate(lowBits, highBits) {
    highBits = ~highBits;
    if (lowBits) {
        lowBits = ~lowBits + 1;
    } else {
        // If lowBits is 0, then bitwise-not is 0xFFFFFFFF,
        // adding 1 to that, results in 0x100000000, which leaves
        // the low bits 0x0 and simply adds one to the high bits.
        highBits += 1;
    }
    return newBits(lowBits, highBits);
}
/**
 * Returns decimal representation of digit1e7 with leading zeros.
 */ const decimalFrom1e7WithLeadingZeros = (digit1e7)=>{
    const partial = String(digit1e7);
    return "0000000".slice(partial.length) + partial;
};
function varint32write(value, bytes) {
    if (value >= 0) {
        // write value as varint 32
        while(value > 0x7f){
            bytes.push(value & 0x7f | 0x80);
            value = value >>> 7;
        }
        bytes.push(value);
    } else {
        for(let i = 0; i < 9; i++){
            bytes.push(value & 127 | 128);
            value = value >> 7;
        }
        bytes.push(1);
    }
}
function varint32read() {
    let b = this.buf[this.pos++];
    let result = b & 0x7f;
    if ((b & 0x80) == 0) {
        this.assertBounds();
        return result;
    }
    b = this.buf[this.pos++];
    result |= (b & 0x7f) << 7;
    if ((b & 0x80) == 0) {
        this.assertBounds();
        return result;
    }
    b = this.buf[this.pos++];
    result |= (b & 0x7f) << 14;
    if ((b & 0x80) == 0) {
        this.assertBounds();
        return result;
    }
    b = this.buf[this.pos++];
    result |= (b & 0x7f) << 21;
    if ((b & 0x80) == 0) {
        this.assertBounds();
        return result;
    }
    // Extract only last 4 bits
    b = this.buf[this.pos++];
    result |= (b & 0x0f) << 28;
    for(let readBytes = 5; (b & 0x80) !== 0 && readBytes < 10; readBytes++)b = this.buf[this.pos++];
    if ((b & 0x80) != 0) throw new Error("invalid varint");
    this.assertBounds();
    // Result can have 32 bits, convert it to unsigned
    return result >>> 0;
}
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/proto-int64.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "protoInt64": (()=>protoInt64)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/assert.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$google$2f$varint$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/google/varint.js [app-route] (ecmascript)");
;
;
function makeInt64Support() {
    const dv = new DataView(new ArrayBuffer(8));
    // note that Safari 14 implements BigInt, but not the DataView methods
    const ok = typeof BigInt === "function" && typeof dv.getBigInt64 === "function" && typeof dv.getBigUint64 === "function" && typeof dv.setBigInt64 === "function" && typeof dv.setBigUint64 === "function" && (typeof process != "object" || typeof process.env != "object" || process.env.BUF_BIGINT_DISABLE !== "1");
    if (ok) {
        const MIN = BigInt("-9223372036854775808"), MAX = BigInt("9223372036854775807"), UMIN = BigInt("0"), UMAX = BigInt("18446744073709551615");
        return {
            zero: BigInt(0),
            supported: true,
            parse (value) {
                const bi = typeof value == "bigint" ? value : BigInt(value);
                if (bi > MAX || bi < MIN) {
                    throw new Error(`int64 invalid: ${value}`);
                }
                return bi;
            },
            uParse (value) {
                const bi = typeof value == "bigint" ? value : BigInt(value);
                if (bi > UMAX || bi < UMIN) {
                    throw new Error(`uint64 invalid: ${value}`);
                }
                return bi;
            },
            enc (value) {
                dv.setBigInt64(0, this.parse(value), true);
                return {
                    lo: dv.getInt32(0, true),
                    hi: dv.getInt32(4, true)
                };
            },
            uEnc (value) {
                dv.setBigInt64(0, this.uParse(value), true);
                return {
                    lo: dv.getInt32(0, true),
                    hi: dv.getInt32(4, true)
                };
            },
            dec (lo, hi) {
                dv.setInt32(0, lo, true);
                dv.setInt32(4, hi, true);
                return dv.getBigInt64(0, true);
            },
            uDec (lo, hi) {
                dv.setInt32(0, lo, true);
                dv.setInt32(4, hi, true);
                return dv.getBigUint64(0, true);
            }
        };
    }
    const assertInt64String = (value)=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(/^-?[0-9]+$/.test(value), `int64 invalid: ${value}`);
    const assertUInt64String = (value)=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(/^[0-9]+$/.test(value), `uint64 invalid: ${value}`);
    return {
        zero: "0",
        supported: false,
        parse (value) {
            if (typeof value != "string") {
                value = value.toString();
            }
            assertInt64String(value);
            return value;
        },
        uParse (value) {
            if (typeof value != "string") {
                value = value.toString();
            }
            assertUInt64String(value);
            return value;
        },
        enc (value) {
            if (typeof value != "string") {
                value = value.toString();
            }
            assertInt64String(value);
            return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$google$2f$varint$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["int64FromString"])(value);
        },
        uEnc (value) {
            if (typeof value != "string") {
                value = value.toString();
            }
            assertUInt64String(value);
            return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$google$2f$varint$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["int64FromString"])(value);
        },
        dec (lo, hi) {
            return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$google$2f$varint$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["int64ToString"])(lo, hi);
        },
        uDec (lo, hi) {
            return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$google$2f$varint$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["uInt64ToString"])(lo, hi);
        }
    };
}
const protoInt64 = makeInt64Support();
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/scalar.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * Scalar value types. This is a subset of field types declared by protobuf
 * enum google.protobuf.FieldDescriptorProto.Type The types GROUP and MESSAGE
 * are omitted, but the numerical values are identical.
 */ __turbopack_context__.s({
    "LongType": (()=>LongType),
    "ScalarType": (()=>ScalarType)
});
var ScalarType;
(function(ScalarType) {
    // 0 is reserved for errors.
    // Order is weird for historical reasons.
    ScalarType[ScalarType["DOUBLE"] = 1] = "DOUBLE";
    ScalarType[ScalarType["FLOAT"] = 2] = "FLOAT";
    // Not ZigZag encoded.  Negative numbers take 10 bytes.  Use TYPE_SINT64 if
    // negative values are likely.
    ScalarType[ScalarType["INT64"] = 3] = "INT64";
    ScalarType[ScalarType["UINT64"] = 4] = "UINT64";
    // Not ZigZag encoded.  Negative numbers take 10 bytes.  Use TYPE_SINT32 if
    // negative values are likely.
    ScalarType[ScalarType["INT32"] = 5] = "INT32";
    ScalarType[ScalarType["FIXED64"] = 6] = "FIXED64";
    ScalarType[ScalarType["FIXED32"] = 7] = "FIXED32";
    ScalarType[ScalarType["BOOL"] = 8] = "BOOL";
    ScalarType[ScalarType["STRING"] = 9] = "STRING";
    // Tag-delimited aggregate.
    // Group type is deprecated and not supported in proto3. However, Proto3
    // implementations should still be able to parse the group wire format and
    // treat group fields as unknown fields.
    // TYPE_GROUP = 10,
    // TYPE_MESSAGE = 11,  // Length-delimited aggregate.
    // New in version 2.
    ScalarType[ScalarType["BYTES"] = 12] = "BYTES";
    ScalarType[ScalarType["UINT32"] = 13] = "UINT32";
    // TYPE_ENUM = 14,
    ScalarType[ScalarType["SFIXED32"] = 15] = "SFIXED32";
    ScalarType[ScalarType["SFIXED64"] = 16] = "SFIXED64";
    ScalarType[ScalarType["SINT32"] = 17] = "SINT32";
    ScalarType[ScalarType["SINT64"] = 18] = "SINT64";
})(ScalarType || (ScalarType = {}));
var LongType;
(function(LongType) {
    /**
     * Use JavaScript BigInt.
     */ LongType[LongType["BIGINT"] = 0] = "BIGINT";
    /**
     * Use JavaScript String.
     *
     * Field option `[jstype = JS_STRING]`.
     */ LongType[LongType["STRING"] = 1] = "STRING";
})(LongType || (LongType = {}));
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/scalars.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "isScalarZeroValue": (()=>isScalarZeroValue),
    "scalarEquals": (()=>scalarEquals),
    "scalarZeroValue": (()=>scalarZeroValue)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/proto-int64.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/scalar.js [app-route] (ecmascript)");
;
;
function scalarEquals(type, a, b) {
    if (a === b) {
        // This correctly matches equal values except BYTES and (possibly) 64-bit integers.
        return true;
    }
    // Special case BYTES - we need to compare each byte individually
    if (type == __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BYTES) {
        if (!(a instanceof Uint8Array) || !(b instanceof Uint8Array)) {
            return false;
        }
        if (a.length !== b.length) {
            return false;
        }
        for(let i = 0; i < a.length; i++){
            if (a[i] !== b[i]) {
                return false;
            }
        }
        return true;
    }
    // Special case 64-bit integers - we support number, string and bigint representation.
    // eslint-disable-next-line @typescript-eslint/switch-exhaustiveness-check
    switch(type){
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].UINT64:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FIXED64:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].INT64:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SFIXED64:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SINT64:
            // Loose comparison will match between 0n, 0 and "0".
            return a == b;
    }
    // Anything that hasn't been caught by strict comparison or special cased
    // BYTES and 64-bit integers is not equal.
    return false;
}
function scalarZeroValue(type, longType) {
    switch(type){
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BOOL:
            return false;
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].UINT64:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FIXED64:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].INT64:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SFIXED64:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SINT64:
            // eslint-disable-next-line @typescript-eslint/no-unsafe-enum-comparison -- acceptable since it's covered by tests
            return longType == 0 ? __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoInt64"].zero : "0";
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].DOUBLE:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FLOAT:
            return 0.0;
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BYTES:
            return new Uint8Array(0);
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].STRING:
            return "";
        default:
            // Handles INT32, UINT32, SINT32, FIXED32, SFIXED32.
            // We do not use individual cases to save a few bytes code size.
            return 0;
    }
}
function isScalarZeroValue(type, value) {
    switch(type){
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BOOL:
            return value === false;
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].STRING:
            return value === "";
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BYTES:
            return value instanceof Uint8Array && !value.byteLength;
        default:
            return value == 0; // Loose comparison matches 0n, 0 and "0"
    }
}
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/extensions.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "createExtensionContainer": (()=>createExtensionContainer),
    "filterUnknownFields": (()=>filterUnknownFields),
    "makeExtension": (()=>makeExtension)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/scalars.js [app-route] (ecmascript)");
;
;
function makeExtension(runtime, typeName, extendee, field) {
    let fi;
    return {
        typeName,
        extendee,
        get field () {
            if (!fi) {
                const i = typeof field == "function" ? field() : field;
                i.name = typeName.split(".").pop();
                i.jsonName = `[${typeName}]`;
                fi = runtime.util.newFieldList([
                    i
                ]).list()[0];
            }
            return fi;
        },
        runtime
    };
}
function createExtensionContainer(extension) {
    const localName = extension.field.localName;
    const container = Object.create(null);
    container[localName] = initExtensionField(extension);
    return [
        container,
        ()=>container[localName]
    ];
}
function initExtensionField(ext) {
    const field = ext.field;
    if (field.repeated) {
        return [];
    }
    if (field.default !== undefined) {
        return field.default;
    }
    switch(field.kind){
        case "enum":
            return field.T.values[0].no;
        case "scalar":
            return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["scalarZeroValue"])(field.T, field.L);
        case "message":
            // eslint-disable-next-line no-case-declarations
            const T = field.T, value = new T();
            return T.fieldWrapper ? T.fieldWrapper.unwrapField(value) : value;
        case "map":
            throw "map fields are not allowed to be extensions";
    }
}
function filterUnknownFields(unknownFields, field) {
    if (!field.repeated && (field.kind == "enum" || field.kind == "scalar")) {
        // singular scalar fields do not merge, we pick the last
        for(let i = unknownFields.length - 1; i >= 0; --i){
            if (unknownFields[i].no == field.no) {
                return [
                    unknownFields[i]
                ];
            }
        }
        return [];
    }
    return unknownFields.filter((uf)=>uf.no === field.no);
}
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/proto-base64.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/* eslint-disable @typescript-eslint/ban-ts-comment, @typescript-eslint/no-unnecessary-condition, prefer-const */ // lookup table from base64 character to byte
__turbopack_context__.s({
    "protoBase64": (()=>protoBase64)
});
let encTable = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".split("");
// lookup table from base64 character *code* to byte because lookup by number is fast
let decTable = [];
for(let i = 0; i < encTable.length; i++)decTable[encTable[i].charCodeAt(0)] = i;
// support base64url variants
decTable["-".charCodeAt(0)] = encTable.indexOf("+");
decTable["_".charCodeAt(0)] = encTable.indexOf("/");
const protoBase64 = {
    /**
     * Decodes a base64 string to a byte array.
     *
     * - ignores white-space, including line breaks and tabs
     * - allows inner padding (can decode concatenated base64 strings)
     * - does not require padding
     * - understands base64url encoding:
     *   "-" instead of "+",
     *   "_" instead of "/",
     *   no padding
     */ dec (base64Str) {
        // estimate byte size, not accounting for inner padding and whitespace
        let es = base64Str.length * 3 / 4;
        if (base64Str[base64Str.length - 2] == "=") es -= 2;
        else if (base64Str[base64Str.length - 1] == "=") es -= 1;
        let bytes = new Uint8Array(es), bytePos = 0, groupPos = 0, b, p = 0; // previous byte
        for(let i = 0; i < base64Str.length; i++){
            b = decTable[base64Str.charCodeAt(i)];
            if (b === undefined) {
                switch(base64Str[i]){
                    // @ts-ignore TS7029: Fallthrough case in switch
                    case "=":
                        groupPos = 0; // reset state when padding found
                    // @ts-ignore TS7029: Fallthrough case in switch
                    case "\n":
                    case "\r":
                    case "\t":
                    case " ":
                        continue; // skip white-space, and padding
                    default:
                        throw Error("invalid base64 string.");
                }
            }
            switch(groupPos){
                case 0:
                    p = b;
                    groupPos = 1;
                    break;
                case 1:
                    bytes[bytePos++] = p << 2 | (b & 48) >> 4;
                    p = b;
                    groupPos = 2;
                    break;
                case 2:
                    bytes[bytePos++] = (p & 15) << 4 | (b & 60) >> 2;
                    p = b;
                    groupPos = 3;
                    break;
                case 3:
                    bytes[bytePos++] = (p & 3) << 6 | b;
                    groupPos = 0;
                    break;
            }
        }
        if (groupPos == 1) throw Error("invalid base64 string.");
        return bytes.subarray(0, bytePos);
    },
    /**
     * Encode a byte array to a base64 string.
     */ enc (bytes) {
        let base64 = "", groupPos = 0, b, p = 0; // carry over from previous byte
        for(let i = 0; i < bytes.length; i++){
            b = bytes[i];
            switch(groupPos){
                case 0:
                    base64 += encTable[b >> 2];
                    p = (b & 3) << 4;
                    groupPos = 1;
                    break;
                case 1:
                    base64 += encTable[p | b >> 4];
                    p = (b & 15) << 2;
                    groupPos = 2;
                    break;
                case 2:
                    base64 += encTable[p | b >> 6];
                    base64 += encTable[b & 63];
                    groupPos = 0;
                    break;
            }
        }
        // add output padding
        if (groupPos) {
            base64 += encTable[p];
            base64 += "=";
            if (groupPos == 1) base64 += "=";
        }
        return base64;
    }
};
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/extension-accessor.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "clearExtension": (()=>clearExtension),
    "getExtension": (()=>getExtension),
    "hasExtension": (()=>hasExtension),
    "setExtension": (()=>setExtension)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/assert.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$extensions$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/extensions.js [app-route] (ecmascript)");
;
;
function getExtension(message, extension, options) {
    assertExtendee(extension, message);
    const opt = extension.runtime.bin.makeReadOptions(options);
    const ufs = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$extensions$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["filterUnknownFields"])(message.getType().runtime.bin.listUnknownFields(message), extension.field);
    const [container, get] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$extensions$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["createExtensionContainer"])(extension);
    for (const uf of ufs){
        extension.runtime.bin.readField(container, opt.readerFactory(uf.data), extension.field, uf.wireType, opt);
    }
    return get();
}
function setExtension(message, extension, value, options) {
    assertExtendee(extension, message);
    const readOpt = extension.runtime.bin.makeReadOptions(options);
    const writeOpt = extension.runtime.bin.makeWriteOptions(options);
    if (hasExtension(message, extension)) {
        const ufs = message.getType().runtime.bin.listUnknownFields(message).filter((uf)=>uf.no != extension.field.no);
        message.getType().runtime.bin.discardUnknownFields(message);
        for (const uf of ufs){
            message.getType().runtime.bin.onUnknownField(message, uf.no, uf.wireType, uf.data);
        }
    }
    const writer = writeOpt.writerFactory();
    let f = extension.field;
    // Implicit presence does not apply to extensions, see https://github.com/protocolbuffers/protobuf/issues/8234
    // We patch the field info to use explicit presence:
    if (!f.opt && !f.repeated && (f.kind == "enum" || f.kind == "scalar")) {
        f = Object.assign(Object.assign({}, extension.field), {
            opt: true
        });
    }
    extension.runtime.bin.writeField(f, value, writer, writeOpt);
    const reader = readOpt.readerFactory(writer.finish());
    while(reader.pos < reader.len){
        const [no, wireType] = reader.tag();
        const data = reader.skip(wireType, no);
        message.getType().runtime.bin.onUnknownField(message, no, wireType, data);
    }
}
function clearExtension(message, extension) {
    assertExtendee(extension, message);
    if (hasExtension(message, extension)) {
        const bin = message.getType().runtime.bin;
        const ufs = bin.listUnknownFields(message).filter((uf)=>uf.no != extension.field.no);
        bin.discardUnknownFields(message);
        for (const uf of ufs){
            bin.onUnknownField(message, uf.no, uf.wireType, uf.data);
        }
    }
}
function hasExtension(message, extension) {
    const messageType = message.getType();
    return extension.extendee.typeName === messageType.typeName && !!messageType.runtime.bin.listUnknownFields(message).find((uf)=>uf.no == extension.field.no);
}
function assertExtendee(extension, message) {
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(extension.extendee.typeName == message.getType().typeName, `extension ${extension.typeName} can only be applied to message ${extension.extendee.typeName}`);
}
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/reflect.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "clearField": (()=>clearField),
    "isFieldSet": (()=>isFieldSet)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/scalars.js [app-route] (ecmascript)");
;
function isFieldSet(field, target) {
    const localName = field.localName;
    if (field.repeated) {
        return target[localName].length > 0;
    }
    if (field.oneof) {
        return target[field.oneof.localName].case === localName; // eslint-disable-line @typescript-eslint/no-unsafe-member-access
    }
    switch(field.kind){
        case "enum":
        case "scalar":
            if (field.opt || field.req) {
                // explicit presence
                return target[localName] !== undefined;
            }
            // implicit presence
            if (field.kind == "enum") {
                return target[localName] !== field.T.values[0].no;
            }
            return !(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["isScalarZeroValue"])(field.T, target[localName]);
        case "message":
            return target[localName] !== undefined;
        case "map":
            return Object.keys(target[localName]).length > 0; // eslint-disable-line @typescript-eslint/no-unsafe-argument
    }
}
function clearField(field, target) {
    const localName = field.localName;
    const implicitPresence = !field.opt && !field.req;
    if (field.repeated) {
        target[localName] = [];
    } else if (field.oneof) {
        target[field.oneof.localName] = {
            case: undefined
        };
    } else {
        switch(field.kind){
            case "map":
                target[localName] = {};
                break;
            case "enum":
                target[localName] = implicitPresence ? field.T.values[0].no : undefined;
                break;
            case "scalar":
                target[localName] = implicitPresence ? (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["scalarZeroValue"])(field.T, field.L) : undefined;
                break;
            case "message":
                target[localName] = undefined;
                break;
        }
    }
}
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/is-message.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "isMessage": (()=>isMessage)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/message.js [app-route] (ecmascript)");
;
function isMessage(arg, type) {
    if (arg === null || typeof arg != "object") {
        return false;
    }
    if (!Object.getOwnPropertyNames(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["Message"].prototype).every((m)=>m in arg && typeof arg[m] == "function")) {
        return false;
    }
    const actualType = arg.getType();
    if (actualType === null || typeof actualType != "function" || !("typeName" in actualType) || typeof actualType.typeName != "string") {
        return false;
    }
    return type === undefined ? true : actualType.typeName == type.typeName;
}
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/field-wrapper.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "getUnwrappedFieldType": (()=>getUnwrappedFieldType),
    "wrapField": (()=>wrapField)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/scalar.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$is$2d$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/is-message.js [app-route] (ecmascript)");
;
;
;
function wrapField(type, value) {
    if ((0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$is$2d$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["isMessage"])(value) || !type.fieldWrapper) {
        return value;
    }
    return type.fieldWrapper.wrapField(value);
}
function getUnwrappedFieldType(field) {
    if (field.fieldKind !== "message") {
        return undefined;
    }
    if (field.repeated) {
        return undefined;
    }
    if (field.oneof != undefined) {
        return undefined;
    }
    return wktWrapperToScalarType[field.message.typeName];
}
const wktWrapperToScalarType = {
    "google.protobuf.DoubleValue": __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].DOUBLE,
    "google.protobuf.FloatValue": __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FLOAT,
    "google.protobuf.Int64Value": __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].INT64,
    "google.protobuf.UInt64Value": __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].UINT64,
    "google.protobuf.Int32Value": __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].INT32,
    "google.protobuf.UInt32Value": __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].UINT32,
    "google.protobuf.BoolValue": __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BOOL,
    "google.protobuf.StringValue": __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].STRING,
    "google.protobuf.BytesValue": __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BYTES
};
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/json-format.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "makeJsonFormat": (()=>makeJsonFormat)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/assert.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/proto-int64.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$base64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/proto-base64.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$extensions$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/extensions.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$extension$2d$accessor$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/extension-accessor.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$reflect$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/reflect.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$field$2d$wrapper$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/field-wrapper.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/scalars.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/scalar.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$is$2d$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/is-message.js [app-route] (ecmascript)");
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
/* eslint-disable no-case-declarations,@typescript-eslint/no-unsafe-argument,@typescript-eslint/no-unsafe-assignment,@typescript-eslint/no-unsafe-member-access,@typescript-eslint/no-unsafe-call */ // Default options for parsing JSON.
const jsonReadDefaults = {
    ignoreUnknownFields: false
};
// Default options for serializing to JSON.
const jsonWriteDefaults = {
    emitDefaultValues: false,
    enumAsInteger: false,
    useProtoFieldName: false,
    prettySpaces: 0
};
function makeReadOptions(options) {
    return options ? Object.assign(Object.assign({}, jsonReadDefaults), options) : jsonReadDefaults;
}
function makeWriteOptions(options) {
    return options ? Object.assign(Object.assign({}, jsonWriteDefaults), options) : jsonWriteDefaults;
}
const tokenNull = Symbol();
const tokenIgnoredUnknownEnum = Symbol();
function makeJsonFormat() {
    return {
        makeReadOptions,
        makeWriteOptions,
        readMessage (type, json, options, message) {
            if (json == null || Array.isArray(json) || typeof json != "object") {
                throw new Error(`cannot decode message ${type.typeName} from JSON: ${debugJsonValue(json)}`);
            }
            message = message !== null && message !== void 0 ? message : new type();
            const oneofSeen = new Map();
            const registry = options.typeRegistry;
            for (const [jsonKey, jsonValue] of Object.entries(json)){
                const field = type.fields.findJsonName(jsonKey);
                if (field) {
                    if (field.oneof) {
                        if (jsonValue === null && field.kind == "scalar") {
                            continue;
                        }
                        const seen = oneofSeen.get(field.oneof);
                        if (seen !== undefined) {
                            throw new Error(`cannot decode message ${type.typeName} from JSON: multiple keys for oneof "${field.oneof.name}" present: "${seen}", "${jsonKey}"`);
                        }
                        oneofSeen.set(field.oneof, jsonKey);
                    }
                    readField(message, jsonValue, field, options, type);
                } else {
                    let found = false;
                    if ((registry === null || registry === void 0 ? void 0 : registry.findExtension) && jsonKey.startsWith("[") && jsonKey.endsWith("]")) {
                        const ext = registry.findExtension(jsonKey.substring(1, jsonKey.length - 1));
                        if (ext && ext.extendee.typeName == type.typeName) {
                            found = true;
                            const [container, get] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$extensions$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["createExtensionContainer"])(ext);
                            readField(container, jsonValue, ext.field, options, ext);
                            // We pass on the options as BinaryReadOptions/BinaryWriteOptions,
                            // so that users can bring their own binary reader and writer factories
                            // if necessary.
                            (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$extension$2d$accessor$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["setExtension"])(message, ext, get(), options);
                        }
                    }
                    if (!found && !options.ignoreUnknownFields) {
                        throw new Error(`cannot decode message ${type.typeName} from JSON: key "${jsonKey}" is unknown`);
                    }
                }
            }
            return message;
        },
        writeMessage (message, options) {
            const type = message.getType();
            const json = {};
            let field;
            try {
                for (field of type.fields.byNumber()){
                    if (!(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$reflect$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["isFieldSet"])(field, message)) {
                        // eslint-disable-next-line @typescript-eslint/strict-boolean-expressions
                        if (field.req) {
                            throw `required field not set`;
                        }
                        if (!options.emitDefaultValues) {
                            continue;
                        }
                        if (!canEmitFieldDefaultValue(field)) {
                            continue;
                        }
                    }
                    const value = field.oneof ? message[field.oneof.localName].value : message[field.localName];
                    const jsonValue = writeField(field, value, options);
                    if (jsonValue !== undefined) {
                        json[options.useProtoFieldName ? field.name : field.jsonName] = jsonValue;
                    }
                }
                const registry = options.typeRegistry;
                if (registry === null || registry === void 0 ? void 0 : registry.findExtensionFor) {
                    for (const uf of type.runtime.bin.listUnknownFields(message)){
                        const ext = registry.findExtensionFor(type.typeName, uf.no);
                        if (ext && (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$extension$2d$accessor$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["hasExtension"])(message, ext)) {
                            // We pass on the options as BinaryReadOptions, so that users can bring their own
                            // binary reader factory if necessary.
                            const value = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$extension$2d$accessor$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["getExtension"])(message, ext, options);
                            const jsonValue = writeField(ext.field, value, options);
                            if (jsonValue !== undefined) {
                                json[ext.field.jsonName] = jsonValue;
                            }
                        }
                    }
                }
            } catch (e) {
                const m = field ? `cannot encode field ${type.typeName}.${field.name} to JSON` : `cannot encode message ${type.typeName} to JSON`;
                const r = e instanceof Error ? e.message : String(e);
                throw new Error(m + (r.length > 0 ? `: ${r}` : ""));
            }
            return json;
        },
        readScalar (type, json, longType) {
            // The signature of our internal function has changed. For backwards-
            // compatibility, we support the old form that is part of the public API
            // through the interface JsonFormat.
            return readScalar(type, json, longType !== null && longType !== void 0 ? longType : __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["LongType"].BIGINT, true);
        },
        writeScalar (type, value, emitDefaultValues) {
            // The signature of our internal function has changed. For backwards-
            // compatibility, we support the old form that is part of the public API
            // through the interface JsonFormat.
            if (value === undefined) {
                return undefined;
            }
            if (emitDefaultValues || (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["isScalarZeroValue"])(type, value)) {
                return writeScalar(type, value);
            }
            return undefined;
        },
        debug: debugJsonValue
    };
}
function debugJsonValue(json) {
    if (json === null) {
        return "null";
    }
    switch(typeof json){
        case "object":
            return Array.isArray(json) ? "array" : "object";
        case "string":
            return json.length > 100 ? "string" : `"${json.split('"').join('\\"')}"`;
        default:
            return String(json);
    }
}
// Read a JSON value for a field.
// The "parentType" argument is only used to provide context in errors.
function readField(target, jsonValue, field, options, parentType) {
    let localName = field.localName;
    if (field.repeated) {
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(field.kind != "map");
        if (jsonValue === null) {
            return;
        }
        if (!Array.isArray(jsonValue)) {
            throw new Error(`cannot decode field ${parentType.typeName}.${field.name} from JSON: ${debugJsonValue(jsonValue)}`);
        }
        const targetArray = target[localName];
        for (const jsonItem of jsonValue){
            if (jsonItem === null) {
                throw new Error(`cannot decode field ${parentType.typeName}.${field.name} from JSON: ${debugJsonValue(jsonItem)}`);
            }
            switch(field.kind){
                case "message":
                    targetArray.push(field.T.fromJson(jsonItem, options));
                    break;
                case "enum":
                    const enumValue = readEnum(field.T, jsonItem, options.ignoreUnknownFields, true);
                    if (enumValue !== tokenIgnoredUnknownEnum) {
                        targetArray.push(enumValue);
                    }
                    break;
                case "scalar":
                    try {
                        targetArray.push(readScalar(field.T, jsonItem, field.L, true));
                    } catch (e) {
                        let m = `cannot decode field ${parentType.typeName}.${field.name} from JSON: ${debugJsonValue(jsonItem)}`;
                        if (e instanceof Error && e.message.length > 0) {
                            m += `: ${e.message}`;
                        }
                        throw new Error(m);
                    }
                    break;
            }
        }
    } else if (field.kind == "map") {
        if (jsonValue === null) {
            return;
        }
        if (typeof jsonValue != "object" || Array.isArray(jsonValue)) {
            throw new Error(`cannot decode field ${parentType.typeName}.${field.name} from JSON: ${debugJsonValue(jsonValue)}`);
        }
        const targetMap = target[localName];
        for (const [jsonMapKey, jsonMapValue] of Object.entries(jsonValue)){
            if (jsonMapValue === null) {
                throw new Error(`cannot decode field ${parentType.typeName}.${field.name} from JSON: map value null`);
            }
            let key;
            try {
                key = readMapKey(field.K, jsonMapKey);
            } catch (e) {
                let m = `cannot decode map key for field ${parentType.typeName}.${field.name} from JSON: ${debugJsonValue(jsonValue)}`;
                if (e instanceof Error && e.message.length > 0) {
                    m += `: ${e.message}`;
                }
                throw new Error(m);
            }
            switch(field.V.kind){
                case "message":
                    targetMap[key] = field.V.T.fromJson(jsonMapValue, options);
                    break;
                case "enum":
                    const enumValue = readEnum(field.V.T, jsonMapValue, options.ignoreUnknownFields, true);
                    if (enumValue !== tokenIgnoredUnknownEnum) {
                        targetMap[key] = enumValue;
                    }
                    break;
                case "scalar":
                    try {
                        targetMap[key] = readScalar(field.V.T, jsonMapValue, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["LongType"].BIGINT, true);
                    } catch (e) {
                        let m = `cannot decode map value for field ${parentType.typeName}.${field.name} from JSON: ${debugJsonValue(jsonValue)}`;
                        if (e instanceof Error && e.message.length > 0) {
                            m += `: ${e.message}`;
                        }
                        throw new Error(m);
                    }
                    break;
            }
        }
    } else {
        if (field.oneof) {
            target = target[field.oneof.localName] = {
                case: localName
            };
            localName = "value";
        }
        switch(field.kind){
            case "message":
                const messageType = field.T;
                if (jsonValue === null && messageType.typeName != "google.protobuf.Value") {
                    return;
                }
                let currentValue = target[localName];
                if ((0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$is$2d$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["isMessage"])(currentValue)) {
                    currentValue.fromJson(jsonValue, options);
                } else {
                    target[localName] = currentValue = messageType.fromJson(jsonValue, options);
                    if (messageType.fieldWrapper && !field.oneof) {
                        target[localName] = messageType.fieldWrapper.unwrapField(currentValue);
                    }
                }
                break;
            case "enum":
                const enumValue = readEnum(field.T, jsonValue, options.ignoreUnknownFields, false);
                switch(enumValue){
                    case tokenNull:
                        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$reflect$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["clearField"])(field, target);
                        break;
                    case tokenIgnoredUnknownEnum:
                        break;
                    default:
                        target[localName] = enumValue;
                        break;
                }
                break;
            case "scalar":
                try {
                    const scalarValue = readScalar(field.T, jsonValue, field.L, false);
                    switch(scalarValue){
                        case tokenNull:
                            (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$reflect$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["clearField"])(field, target);
                            break;
                        default:
                            target[localName] = scalarValue;
                            break;
                    }
                } catch (e) {
                    let m = `cannot decode field ${parentType.typeName}.${field.name} from JSON: ${debugJsonValue(jsonValue)}`;
                    if (e instanceof Error && e.message.length > 0) {
                        m += `: ${e.message}`;
                    }
                    throw new Error(m);
                }
                break;
        }
    }
}
function readMapKey(type, json) {
    if (type === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BOOL) {
        // eslint-disable-next-line @typescript-eslint/switch-exhaustiveness-check
        switch(json){
            case "true":
                json = true;
                break;
            case "false":
                json = false;
                break;
        }
    }
    return readScalar(type, json, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["LongType"].BIGINT, true).toString();
}
function readScalar(type, json, longType, nullAsZeroValue) {
    if (json === null) {
        if (nullAsZeroValue) {
            return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["scalarZeroValue"])(type, longType);
        }
        return tokenNull;
    }
    // every valid case in the switch below returns, and every fall
    // through is regarded as a failure.
    switch(type){
        // float, double: JSON value will be a number or one of the special string values "NaN", "Infinity", and "-Infinity".
        // Either numbers or strings are accepted. Exponent notation is also accepted.
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].DOUBLE:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FLOAT:
            if (json === "NaN") return Number.NaN;
            if (json === "Infinity") return Number.POSITIVE_INFINITY;
            if (json === "-Infinity") return Number.NEGATIVE_INFINITY;
            if (json === "") {
                break;
            }
            if (typeof json == "string" && json.trim().length !== json.length) {
                break;
            }
            if (typeof json != "string" && typeof json != "number") {
                break;
            }
            const float = Number(json);
            if (Number.isNaN(float)) {
                break;
            }
            if (!Number.isFinite(float)) {
                break;
            }
            if (type == __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FLOAT) (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assertFloat32"])(float);
            return float;
        // int32, fixed32, uint32: JSON value will be a decimal number. Either numbers or strings are accepted.
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].INT32:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FIXED32:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SFIXED32:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SINT32:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].UINT32:
            let int32;
            if (typeof json == "number") int32 = json;
            else if (typeof json == "string" && json.length > 0) {
                if (json.trim().length === json.length) int32 = Number(json);
            }
            if (int32 === undefined) break;
            if (type == __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].UINT32 || type == __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FIXED32) (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assertUInt32"])(int32);
            else (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assertInt32"])(int32);
            return int32;
        // int64, fixed64, uint64: JSON value will be a decimal string. Either numbers or strings are accepted.
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].INT64:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SFIXED64:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SINT64:
            if (typeof json != "number" && typeof json != "string") break;
            const long = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoInt64"].parse(json);
            // eslint-disable-next-line @typescript-eslint/strict-boolean-expressions
            return longType ? long.toString() : long;
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FIXED64:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].UINT64:
            if (typeof json != "number" && typeof json != "string") break;
            const uLong = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoInt64"].uParse(json);
            // eslint-disable-next-line @typescript-eslint/strict-boolean-expressions
            return longType ? uLong.toString() : uLong;
        // bool:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BOOL:
            if (typeof json !== "boolean") break;
            return json;
        // string:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].STRING:
            if (typeof json !== "string") {
                break;
            }
            // A string must always contain UTF-8 encoded or 7-bit ASCII.
            // We validate with encodeURIComponent, which appears to be the fastest widely available option.
            try {
                encodeURIComponent(json);
            } catch (e) {
                throw new Error("invalid UTF8");
            }
            return json;
        // bytes: JSON value will be the data encoded as a string using standard base64 encoding with paddings.
        // Either standard or URL-safe base64 encoding with/without paddings are accepted.
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BYTES:
            if (json === "") return new Uint8Array(0);
            if (typeof json !== "string") break;
            return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$base64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoBase64"].dec(json);
    }
    throw new Error();
}
function readEnum(type, json, ignoreUnknownFields, nullAsZeroValue) {
    if (json === null) {
        if (type.typeName == "google.protobuf.NullValue") {
            return 0; // google.protobuf.NullValue.NULL_VALUE = 0
        }
        return nullAsZeroValue ? type.values[0].no : tokenNull;
    }
    // eslint-disable-next-line @typescript-eslint/switch-exhaustiveness-check
    switch(typeof json){
        case "number":
            if (Number.isInteger(json)) {
                return json;
            }
            break;
        case "string":
            const value = type.findName(json);
            if (value !== undefined) {
                return value.no;
            }
            if (ignoreUnknownFields) {
                return tokenIgnoredUnknownEnum;
            }
            break;
    }
    throw new Error(`cannot decode enum ${type.typeName} from JSON: ${debugJsonValue(json)}`);
}
// Decide whether an unset field should be emitted with JSON write option `emitDefaultValues`
function canEmitFieldDefaultValue(field) {
    if (field.repeated || field.kind == "map") {
        // maps are {}, repeated fields are []
        return true;
    }
    if (field.oneof) {
        // oneof fields are never emitted
        return false;
    }
    if (field.kind == "message") {
        // singular message field are allowed to emit JSON null, but we do not
        return false;
    }
    // eslint-disable-next-line @typescript-eslint/strict-boolean-expressions
    if (field.opt || field.req) {
        // the field uses explicit presence, so we cannot emit a zero value
        return false;
    }
    return true;
}
function writeField(field, value, options) {
    if (field.kind == "map") {
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(typeof value == "object" && value != null);
        const jsonObj = {};
        const entries = Object.entries(value);
        switch(field.V.kind){
            case "scalar":
                for (const [entryKey, entryValue] of entries){
                    jsonObj[entryKey.toString()] = writeScalar(field.V.T, entryValue); // JSON standard allows only (double quoted) string as property key
                }
                break;
            case "message":
                for (const [entryKey, entryValue] of entries){
                    // JSON standard allows only (double quoted) string as property key
                    jsonObj[entryKey.toString()] = entryValue.toJson(options);
                }
                break;
            case "enum":
                const enumType = field.V.T;
                for (const [entryKey, entryValue] of entries){
                    // JSON standard allows only (double quoted) string as property key
                    jsonObj[entryKey.toString()] = writeEnum(enumType, entryValue, options.enumAsInteger);
                }
                break;
        }
        return options.emitDefaultValues || entries.length > 0 ? jsonObj : undefined;
    }
    if (field.repeated) {
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(Array.isArray(value));
        const jsonArr = [];
        switch(field.kind){
            case "scalar":
                for(let i = 0; i < value.length; i++){
                    jsonArr.push(writeScalar(field.T, value[i]));
                }
                break;
            case "enum":
                for(let i = 0; i < value.length; i++){
                    jsonArr.push(writeEnum(field.T, value[i], options.enumAsInteger));
                }
                break;
            case "message":
                for(let i = 0; i < value.length; i++){
                    jsonArr.push(value[i].toJson(options));
                }
                break;
        }
        return options.emitDefaultValues || jsonArr.length > 0 ? jsonArr : undefined;
    }
    switch(field.kind){
        case "scalar":
            return writeScalar(field.T, value);
        case "enum":
            return writeEnum(field.T, value, options.enumAsInteger);
        case "message":
            return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$field$2d$wrapper$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["wrapField"])(field.T, value).toJson(options);
    }
}
function writeEnum(type, value, enumAsInteger) {
    var _a;
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(typeof value == "number");
    if (type.typeName == "google.protobuf.NullValue") {
        return null;
    }
    if (enumAsInteger) {
        return value;
    }
    const val = type.findNumber(value);
    return (_a = val === null || val === void 0 ? void 0 : val.name) !== null && _a !== void 0 ? _a : value; // if we don't know the enum value, just return the number
}
function writeScalar(type, value) {
    switch(type){
        // int32, fixed32, uint32: JSON value will be a decimal number. Either numbers or strings are accepted.
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].INT32:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SFIXED32:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SINT32:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FIXED32:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].UINT32:
            (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(typeof value == "number");
            return value;
        // float, double: JSON value will be a number or one of the special string values "NaN", "Infinity", and "-Infinity".
        // Either numbers or strings are accepted. Exponent notation is also accepted.
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FLOAT:
        // assertFloat32(value);
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].DOUBLE:
            (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(typeof value == "number");
            if (Number.isNaN(value)) return "NaN";
            if (value === Number.POSITIVE_INFINITY) return "Infinity";
            if (value === Number.NEGATIVE_INFINITY) return "-Infinity";
            return value;
        // string:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].STRING:
            (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(typeof value == "string");
            return value;
        // bool:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BOOL:
            (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(typeof value == "boolean");
            return value;
        // JSON value will be a decimal string. Either numbers or strings are accepted.
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].UINT64:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FIXED64:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].INT64:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SFIXED64:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SINT64:
            (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(typeof value == "bigint" || typeof value == "string" || typeof value == "number");
            return value.toString();
        // bytes: JSON value will be the data encoded as a string using standard base64 encoding with paddings.
        // Either standard or URL-safe base64 encoding with/without paddings are accepted.
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BYTES:
            (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(value instanceof Uint8Array);
            return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$base64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoBase64"].enc(value);
    }
}
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/binary-encoding.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "BinaryReader": (()=>BinaryReader),
    "BinaryWriter": (()=>BinaryWriter),
    "WireType": (()=>WireType)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$google$2f$varint$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/google/varint.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/assert.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/proto-int64.js [app-route] (ecmascript)");
;
;
;
var WireType;
(function(WireType) {
    /**
     * Used for int32, int64, uint32, uint64, sint32, sint64, bool, enum
     */ WireType[WireType["Varint"] = 0] = "Varint";
    /**
     * Used for fixed64, sfixed64, double.
     * Always 8 bytes with little-endian byte order.
     */ WireType[WireType["Bit64"] = 1] = "Bit64";
    /**
     * Used for string, bytes, embedded messages, packed repeated fields
     *
     * Only repeated numeric types (types which use the varint, 32-bit,
     * or 64-bit wire types) can be packed. In proto3, such fields are
     * packed by default.
     */ WireType[WireType["LengthDelimited"] = 2] = "LengthDelimited";
    /**
     * Start of a tag-delimited aggregate, such as a proto2 group, or a message
     * in editions with message_encoding = DELIMITED.
     */ WireType[WireType["StartGroup"] = 3] = "StartGroup";
    /**
     * End of a tag-delimited aggregate.
     */ WireType[WireType["EndGroup"] = 4] = "EndGroup";
    /**
     * Used for fixed32, sfixed32, float.
     * Always 4 bytes with little-endian byte order.
     */ WireType[WireType["Bit32"] = 5] = "Bit32";
})(WireType || (WireType = {}));
class BinaryWriter {
    constructor(textEncoder){
        /**
         * Previous fork states.
         */ this.stack = [];
        this.textEncoder = textEncoder !== null && textEncoder !== void 0 ? textEncoder : new TextEncoder();
        this.chunks = [];
        this.buf = [];
    }
    /**
     * Return all bytes written and reset this writer.
     */ finish() {
        this.chunks.push(new Uint8Array(this.buf)); // flush the buffer
        let len = 0;
        for(let i = 0; i < this.chunks.length; i++)len += this.chunks[i].length;
        let bytes = new Uint8Array(len);
        let offset = 0;
        for(let i = 0; i < this.chunks.length; i++){
            bytes.set(this.chunks[i], offset);
            offset += this.chunks[i].length;
        }
        this.chunks = [];
        return bytes;
    }
    /**
     * Start a new fork for length-delimited data like a message
     * or a packed repeated field.
     *
     * Must be joined later with `join()`.
     */ fork() {
        this.stack.push({
            chunks: this.chunks,
            buf: this.buf
        });
        this.chunks = [];
        this.buf = [];
        return this;
    }
    /**
     * Join the last fork. Write its length and bytes, then
     * return to the previous state.
     */ join() {
        // get chunk of fork
        let chunk = this.finish();
        // restore previous state
        let prev = this.stack.pop();
        if (!prev) throw new Error("invalid state, fork stack empty");
        this.chunks = prev.chunks;
        this.buf = prev.buf;
        // write length of chunk as varint
        this.uint32(chunk.byteLength);
        return this.raw(chunk);
    }
    /**
     * Writes a tag (field number and wire type).
     *
     * Equivalent to `uint32( (fieldNo << 3 | type) >>> 0 )`.
     *
     * Generated code should compute the tag ahead of time and call `uint32()`.
     */ tag(fieldNo, type) {
        return this.uint32((fieldNo << 3 | type) >>> 0);
    }
    /**
     * Write a chunk of raw bytes.
     */ raw(chunk) {
        if (this.buf.length) {
            this.chunks.push(new Uint8Array(this.buf));
            this.buf = [];
        }
        this.chunks.push(chunk);
        return this;
    }
    /**
     * Write a `uint32` value, an unsigned 32 bit varint.
     */ uint32(value) {
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assertUInt32"])(value);
        // write value as varint 32, inlined for speed
        while(value > 0x7f){
            this.buf.push(value & 0x7f | 0x80);
            value = value >>> 7;
        }
        this.buf.push(value);
        return this;
    }
    /**
     * Write a `int32` value, a signed 32 bit varint.
     */ int32(value) {
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assertInt32"])(value);
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$google$2f$varint$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["varint32write"])(value, this.buf);
        return this;
    }
    /**
     * Write a `bool` value, a variant.
     */ bool(value) {
        this.buf.push(value ? 1 : 0);
        return this;
    }
    /**
     * Write a `bytes` value, length-delimited arbitrary data.
     */ bytes(value) {
        this.uint32(value.byteLength); // write length of chunk as varint
        return this.raw(value);
    }
    /**
     * Write a `string` value, length-delimited data converted to UTF-8 text.
     */ string(value) {
        let chunk = this.textEncoder.encode(value);
        this.uint32(chunk.byteLength); // write length of chunk as varint
        return this.raw(chunk);
    }
    /**
     * Write a `float` value, 32-bit floating point number.
     */ float(value) {
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assertFloat32"])(value);
        let chunk = new Uint8Array(4);
        new DataView(chunk.buffer).setFloat32(0, value, true);
        return this.raw(chunk);
    }
    /**
     * Write a `double` value, a 64-bit floating point number.
     */ double(value) {
        let chunk = new Uint8Array(8);
        new DataView(chunk.buffer).setFloat64(0, value, true);
        return this.raw(chunk);
    }
    /**
     * Write a `fixed32` value, an unsigned, fixed-length 32-bit integer.
     */ fixed32(value) {
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assertUInt32"])(value);
        let chunk = new Uint8Array(4);
        new DataView(chunk.buffer).setUint32(0, value, true);
        return this.raw(chunk);
    }
    /**
     * Write a `sfixed32` value, a signed, fixed-length 32-bit integer.
     */ sfixed32(value) {
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assertInt32"])(value);
        let chunk = new Uint8Array(4);
        new DataView(chunk.buffer).setInt32(0, value, true);
        return this.raw(chunk);
    }
    /**
     * Write a `sint32` value, a signed, zigzag-encoded 32-bit varint.
     */ sint32(value) {
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assertInt32"])(value);
        // zigzag encode
        value = (value << 1 ^ value >> 31) >>> 0;
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$google$2f$varint$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["varint32write"])(value, this.buf);
        return this;
    }
    /**
     * Write a `fixed64` value, a signed, fixed-length 64-bit integer.
     */ sfixed64(value) {
        let chunk = new Uint8Array(8), view = new DataView(chunk.buffer), tc = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoInt64"].enc(value);
        view.setInt32(0, tc.lo, true);
        view.setInt32(4, tc.hi, true);
        return this.raw(chunk);
    }
    /**
     * Write a `fixed64` value, an unsigned, fixed-length 64 bit integer.
     */ fixed64(value) {
        let chunk = new Uint8Array(8), view = new DataView(chunk.buffer), tc = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoInt64"].uEnc(value);
        view.setInt32(0, tc.lo, true);
        view.setInt32(4, tc.hi, true);
        return this.raw(chunk);
    }
    /**
     * Write a `int64` value, a signed 64-bit varint.
     */ int64(value) {
        let tc = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoInt64"].enc(value);
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$google$2f$varint$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["varint64write"])(tc.lo, tc.hi, this.buf);
        return this;
    }
    /**
     * Write a `sint64` value, a signed, zig-zag-encoded 64-bit varint.
     */ sint64(value) {
        let tc = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoInt64"].enc(value), // zigzag encode
        sign = tc.hi >> 31, lo = tc.lo << 1 ^ sign, hi = (tc.hi << 1 | tc.lo >>> 31) ^ sign;
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$google$2f$varint$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["varint64write"])(lo, hi, this.buf);
        return this;
    }
    /**
     * Write a `uint64` value, an unsigned 64-bit varint.
     */ uint64(value) {
        let tc = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoInt64"].uEnc(value);
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$google$2f$varint$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["varint64write"])(tc.lo, tc.hi, this.buf);
        return this;
    }
}
class BinaryReader {
    constructor(buf, textDecoder){
        this.varint64 = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$google$2f$varint$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["varint64read"]; // dirty cast for `this`
        /**
         * Read a `uint32` field, an unsigned 32 bit varint.
         */ this.uint32 = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$google$2f$varint$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["varint32read"]; // dirty cast for `this` and access to protected `buf`
        this.buf = buf;
        this.len = buf.length;
        this.pos = 0;
        this.view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
        this.textDecoder = textDecoder !== null && textDecoder !== void 0 ? textDecoder : new TextDecoder();
    }
    /**
     * Reads a tag - field number and wire type.
     */ tag() {
        let tag = this.uint32(), fieldNo = tag >>> 3, wireType = tag & 7;
        if (fieldNo <= 0 || wireType < 0 || wireType > 5) throw new Error("illegal tag: field no " + fieldNo + " wire type " + wireType);
        return [
            fieldNo,
            wireType
        ];
    }
    /**
     * Skip one element and return the skipped data.
     *
     * When skipping StartGroup, provide the tags field number to check for
     * matching field number in the EndGroup tag.
     */ skip(wireType, fieldNo) {
        let start = this.pos;
        switch(wireType){
            case WireType.Varint:
                while(this.buf[this.pos++] & 0x80){
                // ignore
                }
                break;
            // eslint-disable-next-line
            // @ts-ignore TS7029: Fallthrough case in switch
            case WireType.Bit64:
                this.pos += 4;
            // eslint-disable-next-line
            // @ts-ignore TS7029: Fallthrough case in switch
            case WireType.Bit32:
                this.pos += 4;
                break;
            case WireType.LengthDelimited:
                let len = this.uint32();
                this.pos += len;
                break;
            case WireType.StartGroup:
                for(;;){
                    const [fn, wt] = this.tag();
                    if (wt === WireType.EndGroup) {
                        if (fieldNo !== undefined && fn !== fieldNo) {
                            throw new Error("invalid end group tag");
                        }
                        break;
                    }
                    this.skip(wt, fn);
                }
                break;
            default:
                throw new Error("cant skip wire type " + wireType);
        }
        this.assertBounds();
        return this.buf.subarray(start, this.pos);
    }
    /**
     * Throws error if position in byte array is out of range.
     */ assertBounds() {
        if (this.pos > this.len) throw new RangeError("premature EOF");
    }
    /**
     * Read a `int32` field, a signed 32 bit varint.
     */ int32() {
        return this.uint32() | 0;
    }
    /**
     * Read a `sint32` field, a signed, zigzag-encoded 32-bit varint.
     */ sint32() {
        let zze = this.uint32();
        // decode zigzag
        return zze >>> 1 ^ -(zze & 1);
    }
    /**
     * Read a `int64` field, a signed 64-bit varint.
     */ int64() {
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoInt64"].dec(...this.varint64());
    }
    /**
     * Read a `uint64` field, an unsigned 64-bit varint.
     */ uint64() {
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoInt64"].uDec(...this.varint64());
    }
    /**
     * Read a `sint64` field, a signed, zig-zag-encoded 64-bit varint.
     */ sint64() {
        let [lo, hi] = this.varint64();
        // decode zig zag
        let s = -(lo & 1);
        lo = (lo >>> 1 | (hi & 1) << 31) ^ s;
        hi = hi >>> 1 ^ s;
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoInt64"].dec(lo, hi);
    }
    /**
     * Read a `bool` field, a variant.
     */ bool() {
        let [lo, hi] = this.varint64();
        return lo !== 0 || hi !== 0;
    }
    /**
     * Read a `fixed32` field, an unsigned, fixed-length 32-bit integer.
     */ fixed32() {
        return this.view.getUint32((this.pos += 4) - 4, true);
    }
    /**
     * Read a `sfixed32` field, a signed, fixed-length 32-bit integer.
     */ sfixed32() {
        return this.view.getInt32((this.pos += 4) - 4, true);
    }
    /**
     * Read a `fixed64` field, an unsigned, fixed-length 64 bit integer.
     */ fixed64() {
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoInt64"].uDec(this.sfixed32(), this.sfixed32());
    }
    /**
     * Read a `fixed64` field, a signed, fixed-length 64-bit integer.
     */ sfixed64() {
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoInt64"].dec(this.sfixed32(), this.sfixed32());
    }
    /**
     * Read a `float` field, 32-bit floating point number.
     */ float() {
        return this.view.getFloat32((this.pos += 4) - 4, true);
    }
    /**
     * Read a `double` field, a 64-bit floating point number.
     */ double() {
        return this.view.getFloat64((this.pos += 8) - 8, true);
    }
    /**
     * Read a `bytes` field, length-delimited arbitrary data.
     */ bytes() {
        let len = this.uint32(), start = this.pos;
        this.pos += len;
        this.assertBounds();
        return this.buf.subarray(start, start + len);
    }
    /**
     * Read a `string` field, length-delimited data converted to UTF-8 text.
     */ string() {
        return this.textDecoder.decode(this.bytes());
    }
}
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/binary-format.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "makeBinaryFormat": (()=>makeBinaryFormat),
    "writeMapEntry": (()=>writeMapEntry)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$binary$2d$encoding$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/binary-encoding.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$field$2d$wrapper$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/field-wrapper.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/scalars.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/assert.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$reflect$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/reflect.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/scalar.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$is$2d$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/is-message.js [app-route] (ecmascript)");
;
;
;
;
;
;
;
;
/* eslint-disable prefer-const,no-case-declarations,@typescript-eslint/no-explicit-any,@typescript-eslint/no-unsafe-argument,@typescript-eslint/no-unsafe-assignment,@typescript-eslint/no-unsafe-member-access,@typescript-eslint/no-unsafe-call,@typescript-eslint/no-unsafe-return */ const unknownFieldsSymbol = Symbol("@bufbuild/protobuf/unknown-fields");
// Default options for parsing binary data.
const readDefaults = {
    readUnknownFields: true,
    readerFactory: (bytes)=>new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$binary$2d$encoding$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["BinaryReader"](bytes)
};
// Default options for serializing binary data.
const writeDefaults = {
    writeUnknownFields: true,
    writerFactory: ()=>new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$binary$2d$encoding$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["BinaryWriter"]()
};
function makeReadOptions(options) {
    return options ? Object.assign(Object.assign({}, readDefaults), options) : readDefaults;
}
function makeWriteOptions(options) {
    return options ? Object.assign(Object.assign({}, writeDefaults), options) : writeDefaults;
}
function makeBinaryFormat() {
    return {
        makeReadOptions,
        makeWriteOptions,
        listUnknownFields (message) {
            var _a;
            return (_a = message[unknownFieldsSymbol]) !== null && _a !== void 0 ? _a : [];
        },
        discardUnknownFields (message) {
            delete message[unknownFieldsSymbol];
        },
        writeUnknownFields (message, writer) {
            const m = message;
            const c = m[unknownFieldsSymbol];
            if (c) {
                for (const f of c){
                    writer.tag(f.no, f.wireType).raw(f.data);
                }
            }
        },
        onUnknownField (message, no, wireType, data) {
            const m = message;
            if (!Array.isArray(m[unknownFieldsSymbol])) {
                m[unknownFieldsSymbol] = [];
            }
            m[unknownFieldsSymbol].push({
                no,
                wireType,
                data
            });
        },
        readMessage (message, reader, lengthOrEndTagFieldNo, options, delimitedMessageEncoding) {
            const type = message.getType();
            // eslint-disable-next-line @typescript-eslint/strict-boolean-expressions
            const end = delimitedMessageEncoding ? reader.len : reader.pos + lengthOrEndTagFieldNo;
            let fieldNo, wireType;
            while(reader.pos < end){
                [fieldNo, wireType] = reader.tag();
                if (delimitedMessageEncoding === true && wireType == __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$binary$2d$encoding$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["WireType"].EndGroup) {
                    break;
                }
                const field = type.fields.find(fieldNo);
                if (!field) {
                    const data = reader.skip(wireType, fieldNo);
                    if (options.readUnknownFields) {
                        this.onUnknownField(message, fieldNo, wireType, data);
                    }
                    continue;
                }
                readField(message, reader, field, wireType, options);
            }
            if (delimitedMessageEncoding && // eslint-disable-line @typescript-eslint/strict-boolean-expressions
            (wireType != __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$binary$2d$encoding$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["WireType"].EndGroup || fieldNo !== lengthOrEndTagFieldNo)) {
                throw new Error(`invalid end group tag`);
            }
        },
        readField,
        writeMessage (message, writer, options) {
            const type = message.getType();
            for (const field of type.fields.byNumber()){
                if (!(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$reflect$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["isFieldSet"])(field, message)) {
                    if (field.req) {
                        throw new Error(`cannot encode field ${type.typeName}.${field.name} to binary: required field not set`);
                    }
                    continue;
                }
                const value = field.oneof ? message[field.oneof.localName].value : message[field.localName];
                writeField(field, value, writer, options);
            }
            if (options.writeUnknownFields) {
                this.writeUnknownFields(message, writer);
            }
            return writer;
        },
        writeField (field, value, writer, options) {
            // The behavior of our internal function has changed, it does no longer
            // accept `undefined` values for singular scalar and map.
            // For backwards-compatibility, we support the old form that is part of
            // the public API through the interface BinaryFormat.
            if (value === undefined) {
                return undefined;
            }
            writeField(field, value, writer, options);
        }
    };
}
function readField(target, reader, field, wireType, options) {
    let { repeated, localName } = field;
    if (field.oneof) {
        target = target[field.oneof.localName];
        if (target.case != localName) {
            delete target.value;
        }
        target.case = localName;
        localName = "value";
    }
    switch(field.kind){
        case "scalar":
        case "enum":
            const scalarType = field.kind == "enum" ? __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].INT32 : field.T;
            let read = readScalar;
            // eslint-disable-next-line @typescript-eslint/no-unsafe-enum-comparison -- acceptable since it's covered by tests
            if (field.kind == "scalar" && field.L > 0) {
                read = readScalarLTString;
            }
            if (repeated) {
                let arr = target[localName]; // safe to assume presence of array, oneof cannot contain repeated values
                const isPacked = wireType == __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$binary$2d$encoding$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["WireType"].LengthDelimited && scalarType != __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].STRING && scalarType != __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BYTES;
                if (isPacked) {
                    let e = reader.uint32() + reader.pos;
                    while(reader.pos < e){
                        arr.push(read(reader, scalarType));
                    }
                } else {
                    arr.push(read(reader, scalarType));
                }
            } else {
                target[localName] = read(reader, scalarType);
            }
            break;
        case "message":
            const messageType = field.T;
            if (repeated) {
                // safe to assume presence of array, oneof cannot contain repeated values
                target[localName].push(readMessageField(reader, new messageType(), options, field));
            } else {
                if ((0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$is$2d$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["isMessage"])(target[localName])) {
                    readMessageField(reader, target[localName], options, field);
                } else {
                    target[localName] = readMessageField(reader, new messageType(), options, field);
                    if (messageType.fieldWrapper && !field.oneof && !field.repeated) {
                        target[localName] = messageType.fieldWrapper.unwrapField(target[localName]);
                    }
                }
            }
            break;
        case "map":
            let [mapKey, mapVal] = readMapEntry(field, reader, options);
            // safe to assume presence of map object, oneof cannot contain repeated values
            target[localName][mapKey] = mapVal;
            break;
    }
}
// Read a message, avoiding MessageType.fromBinary() to re-use the
// BinaryReadOptions and the IBinaryReader.
function readMessageField(reader, message, options, field) {
    const format = message.getType().runtime.bin;
    const delimited = field === null || field === void 0 ? void 0 : field.delimited;
    format.readMessage(message, reader, delimited ? field.no : reader.uint32(), options, delimited);
    return message;
}
// Read a map field, expecting key field = 1, value field = 2
function readMapEntry(field, reader, options) {
    const length = reader.uint32(), end = reader.pos + length;
    let key, val;
    while(reader.pos < end){
        const [fieldNo] = reader.tag();
        switch(fieldNo){
            case 1:
                key = readScalar(reader, field.K);
                break;
            case 2:
                switch(field.V.kind){
                    case "scalar":
                        val = readScalar(reader, field.V.T);
                        break;
                    case "enum":
                        val = reader.int32();
                        break;
                    case "message":
                        val = readMessageField(reader, new field.V.T(), options, undefined);
                        break;
                }
                break;
        }
    }
    if (key === undefined) {
        key = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["scalarZeroValue"])(field.K, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["LongType"].BIGINT);
    }
    if (typeof key != "string" && typeof key != "number") {
        key = key.toString();
    }
    if (val === undefined) {
        switch(field.V.kind){
            case "scalar":
                val = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["scalarZeroValue"])(field.V.T, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["LongType"].BIGINT);
                break;
            case "enum":
                val = field.V.T.values[0].no;
                break;
            case "message":
                val = new field.V.T();
                break;
        }
    }
    return [
        key,
        val
    ];
}
// Read a scalar value, but return 64 bit integral types (int64, uint64,
// sint64, fixed64, sfixed64) as string instead of bigint.
function readScalarLTString(reader, type) {
    const v = readScalar(reader, type);
    return typeof v == "bigint" ? v.toString() : v;
}
// Does not use scalarTypeInfo() for better performance.
function readScalar(reader, type) {
    switch(type){
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].STRING:
            return reader.string();
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BOOL:
            return reader.bool();
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].DOUBLE:
            return reader.double();
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FLOAT:
            return reader.float();
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].INT32:
            return reader.int32();
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].INT64:
            return reader.int64();
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].UINT64:
            return reader.uint64();
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FIXED64:
            return reader.fixed64();
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BYTES:
            return reader.bytes();
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FIXED32:
            return reader.fixed32();
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SFIXED32:
            return reader.sfixed32();
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SFIXED64:
            return reader.sfixed64();
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SINT64:
            return reader.sint64();
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].UINT32:
            return reader.uint32();
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SINT32:
            return reader.sint32();
    }
}
function writeField(field, value, writer, options) {
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(value !== undefined);
    const repeated = field.repeated;
    switch(field.kind){
        case "scalar":
        case "enum":
            let scalarType = field.kind == "enum" ? __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].INT32 : field.T;
            if (repeated) {
                (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(Array.isArray(value));
                if (field.packed) {
                    writePacked(writer, scalarType, field.no, value);
                } else {
                    for (const item of value){
                        writeScalar(writer, scalarType, field.no, item);
                    }
                }
            } else {
                writeScalar(writer, scalarType, field.no, value);
            }
            break;
        case "message":
            if (repeated) {
                (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(Array.isArray(value));
                for (const item of value){
                    writeMessageField(writer, options, field, item);
                }
            } else {
                writeMessageField(writer, options, field, value);
            }
            break;
        case "map":
            (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(typeof value == "object" && value != null);
            for (const [key, val] of Object.entries(value)){
                writeMapEntry(writer, options, field, key, val);
            }
            break;
    }
}
function writeMapEntry(writer, options, field, key, value) {
    writer.tag(field.no, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$binary$2d$encoding$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["WireType"].LengthDelimited);
    writer.fork();
    // javascript only allows number or string for object properties
    // we convert from our representation to the protobuf type
    let keyValue = key;
    // eslint-disable-next-line @typescript-eslint/switch-exhaustiveness-check -- we deliberately handle just the special cases for map keys
    switch(field.K){
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].INT32:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FIXED32:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].UINT32:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SFIXED32:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SINT32:
            keyValue = Number.parseInt(key);
            break;
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BOOL:
            (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(key == "true" || key == "false");
            keyValue = key == "true";
            break;
    }
    // write key, expecting key field number = 1
    writeScalar(writer, field.K, 1, keyValue);
    // write value, expecting value field number = 2
    switch(field.V.kind){
        case "scalar":
            writeScalar(writer, field.V.T, 2, value);
            break;
        case "enum":
            writeScalar(writer, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].INT32, 2, value);
            break;
        case "message":
            (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(value !== undefined);
            writer.tag(2, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$binary$2d$encoding$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["WireType"].LengthDelimited).bytes(value.toBinary(options));
            break;
    }
    writer.join();
}
// Value must not be undefined
function writeMessageField(writer, options, field, value) {
    const message = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$field$2d$wrapper$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["wrapField"])(field.T, value);
    // eslint-disable-next-line @typescript-eslint/strict-boolean-expressions
    if (field.delimited) writer.tag(field.no, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$binary$2d$encoding$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["WireType"].StartGroup).raw(message.toBinary(options)).tag(field.no, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$binary$2d$encoding$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["WireType"].EndGroup);
    else writer.tag(field.no, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$binary$2d$encoding$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["WireType"].LengthDelimited).bytes(message.toBinary(options));
}
function writeScalar(writer, type, fieldNo, value) {
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(value !== undefined);
    let [wireType, method] = scalarTypeInfo(type);
    writer.tag(fieldNo, wireType)[method](value);
}
function writePacked(writer, type, fieldNo, value) {
    if (!value.length) {
        return;
    }
    writer.tag(fieldNo, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$binary$2d$encoding$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["WireType"].LengthDelimited).fork();
    let [, method] = scalarTypeInfo(type);
    for(let i = 0; i < value.length; i++){
        writer[method](value[i]);
    }
    writer.join();
}
/**
 * Get information for writing a scalar value.
 *
 * Returns tuple:
 * [0]: appropriate WireType
 * [1]: name of the appropriate method of IBinaryWriter
 * [2]: whether the given value is a default value for proto3 semantics
 *
 * If argument `value` is omitted, [2] is always false.
 */ // TODO replace call-sites writeScalar() and writePacked(), then remove
function scalarTypeInfo(type) {
    let wireType = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$binary$2d$encoding$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["WireType"].Varint;
    // eslint-disable-next-line @typescript-eslint/switch-exhaustiveness-check -- INT32, UINT32, SINT32 are covered by the defaults
    switch(type){
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BYTES:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].STRING:
            wireType = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$binary$2d$encoding$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["WireType"].LengthDelimited;
            break;
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].DOUBLE:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FIXED64:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SFIXED64:
            wireType = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$binary$2d$encoding$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["WireType"].Bit64;
            break;
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FIXED32:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].SFIXED32:
        case __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].FLOAT:
            wireType = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$binary$2d$encoding$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["WireType"].Bit32;
            break;
    }
    const method = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"][type].toLowerCase();
    return [
        wireType,
        method
    ];
}
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/util-common.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "makeUtilCommon": (()=>makeUtilCommon)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$enum$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/enum.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/scalars.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/scalar.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$is$2d$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/is-message.js [app-route] (ecmascript)");
;
;
;
;
;
function makeUtilCommon() {
    return {
        setEnumType: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$enum$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["setEnumType"],
        initPartial (source, target) {
            if (source === undefined) {
                return;
            }
            const type = target.getType();
            for (const member of type.fields.byMember()){
                const localName = member.localName, t = target, s = source;
                if (s[localName] == null) {
                    continue;
                }
                switch(member.kind){
                    case "oneof":
                        const sk = s[localName].case;
                        if (sk === undefined) {
                            continue;
                        }
                        const sourceField = member.findField(sk);
                        let val = s[localName].value;
                        if (sourceField && sourceField.kind == "message" && !(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$is$2d$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["isMessage"])(val, sourceField.T)) {
                            val = new sourceField.T(val);
                        } else if (sourceField && sourceField.kind === "scalar" && sourceField.T === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BYTES) {
                            val = toU8Arr(val);
                        }
                        t[localName] = {
                            case: sk,
                            value: val
                        };
                        break;
                    case "scalar":
                    case "enum":
                        let copy = s[localName];
                        if (member.T === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BYTES) {
                            copy = member.repeated ? copy.map(toU8Arr) : toU8Arr(copy);
                        }
                        t[localName] = copy;
                        break;
                    case "map":
                        switch(member.V.kind){
                            case "scalar":
                            case "enum":
                                if (member.V.T === __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BYTES) {
                                    for (const [k, v] of Object.entries(s[localName])){
                                        t[localName][k] = toU8Arr(v);
                                    }
                                } else {
                                    Object.assign(t[localName], s[localName]);
                                }
                                break;
                            case "message":
                                const messageType = member.V.T;
                                for (const k of Object.keys(s[localName])){
                                    let val = s[localName][k];
                                    if (!messageType.fieldWrapper) {
                                        // We only take partial input for messages that are not a wrapper type.
                                        // For those messages, we recursively normalize the partial input.
                                        val = new messageType(val);
                                    }
                                    t[localName][k] = val;
                                }
                                break;
                        }
                        break;
                    case "message":
                        const mt = member.T;
                        if (member.repeated) {
                            t[localName] = s[localName].map((val)=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$is$2d$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["isMessage"])(val, mt) ? val : new mt(val));
                        } else {
                            const val = s[localName];
                            if (mt.fieldWrapper) {
                                if (// We can't use BytesValue.typeName as that will create a circular import
                                mt.typeName === "google.protobuf.BytesValue") {
                                    t[localName] = toU8Arr(val);
                                } else {
                                    t[localName] = val;
                                }
                            } else {
                                t[localName] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$is$2d$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["isMessage"])(val, mt) ? val : new mt(val);
                            }
                        }
                        break;
                }
            }
        },
        // TODO use isFieldSet() here to support future field presence
        equals (type, a, b) {
            if (a === b) {
                return true;
            }
            if (!a || !b) {
                return false;
            }
            return type.fields.byMember().every((m)=>{
                const va = a[m.localName];
                const vb = b[m.localName];
                if (m.repeated) {
                    if (va.length !== vb.length) {
                        return false;
                    }
                    // eslint-disable-next-line @typescript-eslint/switch-exhaustiveness-check -- repeated fields are never "map"
                    switch(m.kind){
                        case "message":
                            return va.every((a, i)=>m.T.equals(a, vb[i]));
                        case "scalar":
                            return va.every((a, i)=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["scalarEquals"])(m.T, a, vb[i]));
                        case "enum":
                            return va.every((a, i)=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["scalarEquals"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].INT32, a, vb[i]));
                    }
                    throw new Error(`repeated cannot contain ${m.kind}`);
                }
                switch(m.kind){
                    case "message":
                        let a1 = va;
                        let b1 = vb;
                        if (m.T.fieldWrapper) {
                            if (a1 !== undefined && !(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$is$2d$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["isMessage"])(a1)) {
                                a1 = m.T.fieldWrapper.wrapField(a1);
                            }
                            if (b1 !== undefined && !(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$is$2d$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["isMessage"])(b1)) {
                                b1 = m.T.fieldWrapper.wrapField(b1);
                            }
                        }
                        return m.T.equals(a1, b1);
                    case "enum":
                        return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["scalarEquals"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].INT32, va, vb);
                    case "scalar":
                        return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["scalarEquals"])(m.T, va, vb);
                    case "oneof":
                        if (va.case !== vb.case) {
                            return false;
                        }
                        const s = m.findField(va.case);
                        if (s === undefined) {
                            return true;
                        }
                        // eslint-disable-next-line @typescript-eslint/switch-exhaustiveness-check -- oneof fields are never "map"
                        switch(s.kind){
                            case "message":
                                return s.T.equals(va.value, vb.value);
                            case "enum":
                                return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["scalarEquals"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].INT32, va.value, vb.value);
                            case "scalar":
                                return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["scalarEquals"])(s.T, va.value, vb.value);
                        }
                        throw new Error(`oneof cannot contain ${s.kind}`);
                    case "map":
                        const keys = Object.keys(va).concat(Object.keys(vb));
                        switch(m.V.kind){
                            case "message":
                                const messageType = m.V.T;
                                return keys.every((k)=>messageType.equals(va[k], vb[k]));
                            case "enum":
                                return keys.every((k)=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["scalarEquals"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].INT32, va[k], vb[k]));
                            case "scalar":
                                const scalarType = m.V.T;
                                return keys.every((k)=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["scalarEquals"])(scalarType, va[k], vb[k]));
                        }
                        break;
                }
            });
        },
        // TODO use isFieldSet() here to support future field presence
        clone (message) {
            const type = message.getType(), target = new type(), any = target;
            for (const member of type.fields.byMember()){
                const source = message[member.localName];
                let copy;
                if (member.repeated) {
                    copy = source.map(cloneSingularField);
                } else if (member.kind == "map") {
                    copy = any[member.localName];
                    for (const [key, v] of Object.entries(source)){
                        copy[key] = cloneSingularField(v);
                    }
                } else if (member.kind == "oneof") {
                    const f = member.findField(source.case);
                    copy = f ? {
                        case: source.case,
                        value: cloneSingularField(source.value)
                    } : {
                        case: undefined
                    };
                } else {
                    copy = cloneSingularField(source);
                }
                any[member.localName] = copy;
            }
            for (const uf of type.runtime.bin.listUnknownFields(message)){
                type.runtime.bin.onUnknownField(any, uf.no, uf.wireType, uf.data);
            }
            return target;
        }
    };
}
// clone a single field value - i.e. the element type of repeated fields, the value type of maps
function cloneSingularField(value) {
    if (value === undefined) {
        return value;
    }
    if ((0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$is$2d$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["isMessage"])(value)) {
        return value.clone();
    }
    if (value instanceof Uint8Array) {
        const c = new Uint8Array(value.byteLength);
        c.set(value);
        return c;
    }
    return value;
}
// converts any ArrayLike<number> to Uint8Array if necessary.
function toU8Arr(input) {
    return input instanceof Uint8Array ? input : new Uint8Array(input);
}
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/proto-runtime.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "makeProtoRuntime": (()=>makeProtoRuntime)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$enum$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/enum.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$message$2d$type$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/message-type.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$extensions$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/extensions.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$json$2d$format$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/json-format.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$binary$2d$format$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/binary-format.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$util$2d$common$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/util-common.js [app-route] (ecmascript)");
;
;
;
;
;
;
function makeProtoRuntime(syntax, newFieldList, initFields) {
    return {
        syntax,
        json: (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$json$2d$format$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["makeJsonFormat"])(),
        bin: (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$binary$2d$format$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["makeBinaryFormat"])(),
        util: Object.assign(Object.assign({}, (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$util$2d$common$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["makeUtilCommon"])()), {
            newFieldList,
            initFields
        }),
        makeMessageType (typeName, fields, opt) {
            return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$message$2d$type$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["makeMessageType"])(this, typeName, fields, opt);
        },
        makeEnum: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$enum$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["makeEnum"],
        makeEnumType: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$enum$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["makeEnumType"],
        getEnumType: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$enum$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["getEnumType"],
        makeExtension (typeName, extendee, field) {
            return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$extensions$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["makeExtension"])(this, typeName, extendee, field);
        }
    };
}
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/field-list.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "InternalFieldList": (()=>InternalFieldList)
});
class InternalFieldList {
    constructor(fields, normalizer){
        this._fields = fields;
        this._normalizer = normalizer;
    }
    findJsonName(jsonName) {
        if (!this.jsonNames) {
            const t = {};
            for (const f of this.list()){
                t[f.jsonName] = t[f.name] = f;
            }
            this.jsonNames = t;
        }
        return this.jsonNames[jsonName];
    }
    find(fieldNo) {
        if (!this.numbers) {
            const t = {};
            for (const f of this.list()){
                t[f.no] = f;
            }
            this.numbers = t;
        }
        return this.numbers[fieldNo];
    }
    list() {
        if (!this.all) {
            this.all = this._normalizer(this._fields);
        }
        return this.all;
    }
    byNumber() {
        if (!this.numbersAsc) {
            this.numbersAsc = this.list().concat().sort((a, b)=>a.no - b.no);
        }
        return this.numbersAsc;
    }
    byMember() {
        if (!this.members) {
            this.members = [];
            const a = this.members;
            let o;
            for (const f of this.list()){
                if (f.oneof) {
                    if (f.oneof !== o) {
                        o = f.oneof;
                        a.push(o);
                    }
                } else {
                    a.push(f);
                }
            }
        }
        return this.members;
    }
}
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/names.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * Returns the name of a protobuf element in generated code.
 *
 * Field names - including oneofs - are converted to lowerCamelCase. For
 * messages, enumerations and services, the package name is stripped from
 * the type name. For nested messages and enumerations, the names are joined
 * with an underscore. For methods, the first character is made lowercase.
 */ __turbopack_context__.s({
    "fieldJsonName": (()=>fieldJsonName),
    "findEnumSharedPrefix": (()=>findEnumSharedPrefix),
    "localFieldName": (()=>localFieldName),
    "localName": (()=>localName),
    "localOneofName": (()=>localOneofName),
    "safeIdentifier": (()=>safeIdentifier),
    "safeObjectProperty": (()=>safeObjectProperty)
});
function localName(desc) {
    switch(desc.kind){
        case "field":
            return localFieldName(desc.name, desc.oneof !== undefined);
        case "oneof":
            return localOneofName(desc.name);
        case "enum":
        case "message":
        case "service":
        case "extension":
            {
                const pkg = desc.file.proto.package;
                const offset = pkg === undefined ? 0 : pkg.length + 1;
                const name = desc.typeName.substring(offset).replace(/\./g, "_");
                // For services, we only care about safe identifiers, not safe object properties,
                // but we have shipped v1 with a bug that respected object properties, and we
                // do not want to introduce a breaking change, so we continue to escape for
                // safe object properties.
                // See https://github.com/bufbuild/protobuf-es/pull/391
                return safeObjectProperty(safeIdentifier(name));
            }
        case "enum_value":
            {
                let name = desc.name;
                const sharedPrefix = desc.parent.sharedPrefix;
                if (sharedPrefix !== undefined) {
                    name = name.substring(sharedPrefix.length);
                }
                return safeObjectProperty(name);
            }
        case "rpc":
            {
                let name = desc.name;
                if (name.length == 0) {
                    return name;
                }
                name = name[0].toLowerCase() + name.substring(1);
                return safeObjectProperty(name);
            }
    }
}
function localFieldName(protoName, inOneof) {
    const name = protoCamelCase(protoName);
    if (inOneof) {
        // oneof member names are not properties, but values of the `case` property.
        return name;
    }
    return safeObjectProperty(safeMessageProperty(name));
}
function localOneofName(protoName) {
    return localFieldName(protoName, false);
}
const fieldJsonName = protoCamelCase;
function findEnumSharedPrefix(enumName, valueNames) {
    const prefix = camelToSnakeCase(enumName) + "_";
    for (const name of valueNames){
        if (!name.toLowerCase().startsWith(prefix)) {
            return undefined;
        }
        const shortName = name.substring(prefix.length);
        if (shortName.length == 0) {
            return undefined;
        }
        if (/^\d/.test(shortName)) {
            // identifiers must not start with numbers
            return undefined;
        }
    }
    return prefix;
}
/**
 * Converts lowerCamelCase or UpperCamelCase into lower_snake_case.
 * This is used to find shared prefixes in an enum.
 */ function camelToSnakeCase(camel) {
    return (camel.substring(0, 1) + camel.substring(1).replace(/[A-Z]/g, (c)=>"_" + c)).toLowerCase();
}
/**
 * Converts snake_case to protoCamelCase according to the convention
 * used by protoc to convert a field name to a JSON name.
 */ function protoCamelCase(snakeCase) {
    let capNext = false;
    const b = [];
    for(let i = 0; i < snakeCase.length; i++){
        let c = snakeCase.charAt(i);
        switch(c){
            case "_":
                capNext = true;
                break;
            case "0":
            case "1":
            case "2":
            case "3":
            case "4":
            case "5":
            case "6":
            case "7":
            case "8":
            case "9":
                b.push(c);
                capNext = false;
                break;
            default:
                if (capNext) {
                    capNext = false;
                    c = c.toUpperCase();
                }
                b.push(c);
                break;
        }
    }
    return b.join("");
}
/**
 * Names that cannot be used for identifiers, such as class names,
 * but _can_ be used for object properties.
 */ const reservedIdentifiers = new Set([
    // ECMAScript 2015 keywords
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "debugger",
    "default",
    "delete",
    "do",
    "else",
    "export",
    "extends",
    "false",
    "finally",
    "for",
    "function",
    "if",
    "import",
    "in",
    "instanceof",
    "new",
    "null",
    "return",
    "super",
    "switch",
    "this",
    "throw",
    "true",
    "try",
    "typeof",
    "var",
    "void",
    "while",
    "with",
    "yield",
    // ECMAScript 2015 future reserved keywords
    "enum",
    "implements",
    "interface",
    "let",
    "package",
    "private",
    "protected",
    "public",
    "static",
    // Class name cannot be 'Object' when targeting ES5 with module CommonJS
    "Object",
    // TypeScript keywords that cannot be used for types (as opposed to variables)
    "bigint",
    "number",
    "boolean",
    "string",
    "object",
    // Identifiers reserved for the runtime, so we can generate legible code
    "globalThis",
    "Uint8Array",
    "Partial"
]);
/**
 * Names that cannot be used for object properties because they are reserved
 * by built-in JavaScript properties.
 */ const reservedObjectProperties = new Set([
    // names reserved by JavaScript
    "constructor",
    "toString",
    "toJSON",
    "valueOf"
]);
/**
 * Names that cannot be used for object properties because they are reserved
 * by the runtime.
 */ const reservedMessageProperties = new Set([
    // names reserved by the runtime
    "getType",
    "clone",
    "equals",
    "fromBinary",
    "fromJson",
    "fromJsonString",
    "toBinary",
    "toJson",
    "toJsonString",
    // names reserved by the runtime for the future
    "toObject"
]);
const fallback = (name)=>`${name}$`;
/**
 * Will wrap names that are Object prototype properties or names reserved
 * for `Message`s.
 */ const safeMessageProperty = (name)=>{
    if (reservedMessageProperties.has(name)) {
        return fallback(name);
    }
    return name;
};
const safeObjectProperty = (name)=>{
    if (reservedObjectProperties.has(name)) {
        return fallback(name);
    }
    return name;
};
const safeIdentifier = (name)=>{
    if (reservedIdentifiers.has(name)) {
        return fallback(name);
    }
    return name;
};
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/field.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "InternalOneofInfo": (()=>InternalOneofInfo)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$names$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/names.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/assert.js [app-route] (ecmascript)");
;
;
class InternalOneofInfo {
    constructor(name){
        this.kind = "oneof";
        this.repeated = false;
        this.packed = false;
        this.opt = false;
        this.req = false;
        this.default = undefined;
        this.fields = [];
        this.name = name;
        this.localName = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$names$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["localOneofName"])(name);
    }
    addField(field) {
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$assert$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["assert"])(field.oneof === this, `field ${field.name} not one of ${this.name}`);
        this.fields.push(field);
    }
    findField(localName) {
        if (!this._lookup) {
            this._lookup = Object.create(null);
            for(let i = 0; i < this.fields.length; i++){
                this._lookup[this.fields[i].localName] = this.fields[i];
            }
        }
        return this._lookup[localName];
    }
}
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/field-normalize.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "normalizeFieldInfos": (()=>normalizeFieldInfos)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$field$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/field.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$names$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/names.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/scalar.js [app-route] (ecmascript)");
;
;
;
function normalizeFieldInfos(fieldInfos, packedByDefault) {
    var _a, _b, _c, _d, _e, _f;
    const r = [];
    let o;
    for (const field of typeof fieldInfos == "function" ? fieldInfos() : fieldInfos){
        const f = field;
        f.localName = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$names$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["localFieldName"])(field.name, field.oneof !== undefined);
        f.jsonName = (_a = field.jsonName) !== null && _a !== void 0 ? _a : (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$names$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["fieldJsonName"])(field.name);
        f.repeated = (_b = field.repeated) !== null && _b !== void 0 ? _b : false;
        if (field.kind == "scalar") {
            f.L = (_c = field.L) !== null && _c !== void 0 ? _c : __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["LongType"].BIGINT;
        }
        f.delimited = (_d = field.delimited) !== null && _d !== void 0 ? _d : false;
        f.req = (_e = field.req) !== null && _e !== void 0 ? _e : false;
        f.opt = (_f = field.opt) !== null && _f !== void 0 ? _f : false;
        if (field.packed === undefined) {
            if (packedByDefault) {
                f.packed = field.kind == "enum" || field.kind == "scalar" && field.T != __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].BYTES && field.T != __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$scalar$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ScalarType"].STRING;
            } else {
                f.packed = false;
            }
        }
        // We do not surface options at this time
        // f.options = field.options ?? emptyReadonlyObject;
        if (field.oneof !== undefined) {
            const ooname = typeof field.oneof == "string" ? field.oneof : field.oneof.name;
            if (!o || o.name != ooname) {
                o = new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$field$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["InternalOneofInfo"](ooname);
            }
            f.oneof = o;
            o.addField(f);
        }
        r.push(f);
    }
    return r;
}
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/proto3.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "proto3": (()=>proto3)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$proto$2d$runtime$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/proto-runtime.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$field$2d$list$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/field-list.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/scalars.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$field$2d$normalize$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/private/field-normalize.js [app-route] (ecmascript)");
;
;
;
;
const proto3 = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$proto$2d$runtime$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["makeProtoRuntime"])("proto3", (fields)=>{
    return new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$field$2d$list$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["InternalFieldList"](fields, (source)=>(0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$field$2d$normalize$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["normalizeFieldInfos"])(source, true));
}, // TODO merge with proto2 and initExtensionField, also see initPartial, equals, clone
(target)=>{
    for (const member of target.getType().fields.byMember()){
        if (member.opt) {
            continue;
        }
        const name = member.localName, t = target;
        if (member.repeated) {
            t[name] = [];
            continue;
        }
        switch(member.kind){
            case "oneof":
                t[name] = {
                    case: undefined
                };
                break;
            case "enum":
                t[name] = 0;
                break;
            case "map":
                t[name] = {};
                break;
            case "scalar":
                t[name] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$private$2f$scalars$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["scalarZeroValue"])(member.T, member.L);
                break;
            case "message":
                break;
        }
    }
});
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/google/protobuf/timestamp_pb.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "Timestamp": (()=>Timestamp)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/message.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/proto-int64.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto3$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/proto3.js [app-route] (ecmascript)");
;
;
;
class Timestamp extends __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["Message"] {
    constructor(data){
        super();
        /**
         * Represents seconds of UTC time since Unix epoch
         * 1970-01-01T00:00:00Z. Must be from 0001-01-01T00:00:00Z to
         * 9999-12-31T23:59:59Z inclusive.
         *
         * @generated from field: int64 seconds = 1;
         */ this.seconds = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoInt64"].zero;
        /**
         * Non-negative fractions of a second at nanosecond resolution. Negative
         * second values with fractions must still have non-negative nanos values
         * that count forward in time. Must be from 0 to 999,999,999
         * inclusive.
         *
         * @generated from field: int32 nanos = 2;
         */ this.nanos = 0;
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto3$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["proto3"].util.initPartial(data, this);
    }
    fromJson(json, options) {
        if (typeof json !== "string") {
            throw new Error(`cannot decode google.protobuf.Timestamp from JSON: ${__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto3$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["proto3"].json.debug(json)}`);
        }
        const matches = json.match(/^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(?:Z|\.([0-9]{3,9})Z|([+-][0-9][0-9]:[0-9][0-9]))$/);
        if (!matches) {
            throw new Error(`cannot decode google.protobuf.Timestamp from JSON: invalid RFC 3339 string`);
        }
        const ms = Date.parse(matches[1] + "-" + matches[2] + "-" + matches[3] + "T" + matches[4] + ":" + matches[5] + ":" + matches[6] + (matches[8] ? matches[8] : "Z"));
        if (Number.isNaN(ms)) {
            throw new Error(`cannot decode google.protobuf.Timestamp from JSON: invalid RFC 3339 string`);
        }
        if (ms < Date.parse("0001-01-01T00:00:00Z") || ms > Date.parse("9999-12-31T23:59:59Z")) {
            throw new Error(`cannot decode message google.protobuf.Timestamp from JSON: must be from 0001-01-01T00:00:00Z to 9999-12-31T23:59:59Z inclusive`);
        }
        this.seconds = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoInt64"].parse(ms / 1000);
        this.nanos = 0;
        if (matches[7]) {
            this.nanos = parseInt("1" + matches[7] + "0".repeat(9 - matches[7].length)) - 1000000000;
        }
        return this;
    }
    toJson(options) {
        const ms = Number(this.seconds) * 1000;
        if (ms < Date.parse("0001-01-01T00:00:00Z") || ms > Date.parse("9999-12-31T23:59:59Z")) {
            throw new Error(`cannot encode google.protobuf.Timestamp to JSON: must be from 0001-01-01T00:00:00Z to 9999-12-31T23:59:59Z inclusive`);
        }
        if (this.nanos < 0) {
            throw new Error(`cannot encode google.protobuf.Timestamp to JSON: nanos must not be negative`);
        }
        let z = "Z";
        if (this.nanos > 0) {
            const nanosStr = (this.nanos + 1000000000).toString().substring(1);
            if (nanosStr.substring(3) === "000000") {
                z = "." + nanosStr.substring(0, 3) + "Z";
            } else if (nanosStr.substring(6) === "000") {
                z = "." + nanosStr.substring(0, 6) + "Z";
            } else {
                z = "." + nanosStr + "Z";
            }
        }
        return new Date(ms).toISOString().replace(".000Z", z);
    }
    toDate() {
        return new Date(Number(this.seconds) * 1000 + Math.ceil(this.nanos / 1000000));
    }
    static now() {
        return Timestamp.fromDate(new Date());
    }
    static fromDate(date) {
        const ms = date.getTime();
        return new Timestamp({
            seconds: __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoInt64"].parse(Math.floor(ms / 1000)),
            nanos: ms % 1000 * 1000000
        });
    }
    static fromBinary(bytes, options) {
        return new Timestamp().fromBinary(bytes, options);
    }
    static fromJson(jsonValue, options) {
        return new Timestamp().fromJson(jsonValue, options);
    }
    static fromJsonString(jsonString, options) {
        return new Timestamp().fromJsonString(jsonString, options);
    }
    static equals(a, b) {
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto3$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["proto3"].util.equals(Timestamp, a, b);
    }
}
Timestamp.runtime = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto3$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["proto3"];
Timestamp.typeName = "google.protobuf.Timestamp";
Timestamp.fields = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto3$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["proto3"].util.newFieldList(()=>[
        {
            no: 1,
            name: "seconds",
            kind: "scalar",
            T: 3 /* ScalarType.INT64 */ 
        },
        {
            no: 2,
            name: "nanos",
            kind: "scalar",
            T: 5 /* ScalarType.INT32 */ 
        }
    ]);
}}),
"[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/google/protobuf/duration_pb.js [app-route] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
// Copyright 2021-2024 Buf Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__turbopack_context__.s({
    "Duration": (()=>Duration)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/message.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/proto-int64.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto3$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/.pnpm/@bufbuild+protobuf@1.10.1/node_modules/@bufbuild/protobuf/dist/esm/proto3.js [app-route] (ecmascript)");
;
;
;
class Duration extends __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$message$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["Message"] {
    constructor(data){
        super();
        /**
         * Signed seconds of the span of time. Must be from -315,576,000,000
         * to +315,576,000,000 inclusive. Note: these bounds are computed from:
         * 60 sec/min * 60 min/hr * 24 hr/day * 365.25 days/year * 10000 years
         *
         * @generated from field: int64 seconds = 1;
         */ this.seconds = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoInt64"].zero;
        /**
         * Signed fractions of a second at nanosecond resolution of the span
         * of time. Durations less than one second are represented with a 0
         * `seconds` field and a positive or negative `nanos` field. For durations
         * of one second or more, a non-zero value for the `nanos` field must be
         * of the same sign as the `seconds` field. Must be from -999,999,999
         * to +999,999,999 inclusive.
         *
         * @generated from field: int32 nanos = 2;
         */ this.nanos = 0;
        __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto3$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["proto3"].util.initPartial(data, this);
    }
    fromJson(json, options) {
        if (typeof json !== "string") {
            throw new Error(`cannot decode google.protobuf.Duration from JSON: ${__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto3$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["proto3"].json.debug(json)}`);
        }
        const match = json.match(/^(-?[0-9]+)(?:\.([0-9]+))?s/);
        if (match === null) {
            throw new Error(`cannot decode google.protobuf.Duration from JSON: ${__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto3$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["proto3"].json.debug(json)}`);
        }
        const longSeconds = Number(match[1]);
        if (longSeconds > 315576000000 || longSeconds < -315576000000) {
            throw new Error(`cannot decode google.protobuf.Duration from JSON: ${__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto3$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["proto3"].json.debug(json)}`);
        }
        this.seconds = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto$2d$int64$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["protoInt64"].parse(longSeconds);
        if (typeof match[2] == "string") {
            const nanosStr = match[2] + "0".repeat(9 - match[2].length);
            this.nanos = parseInt(nanosStr);
            if (longSeconds < 0 || Object.is(longSeconds, -0)) {
                this.nanos = -this.nanos;
            }
        }
        return this;
    }
    toJson(options) {
        if (Number(this.seconds) > 315576000000 || Number(this.seconds) < -315576000000) {
            throw new Error(`cannot encode google.protobuf.Duration to JSON: value out of range`);
        }
        let text = this.seconds.toString();
        if (this.nanos !== 0) {
            let nanosStr = Math.abs(this.nanos).toString();
            nanosStr = "0".repeat(9 - nanosStr.length) + nanosStr;
            if (nanosStr.substring(3) === "000000") {
                nanosStr = nanosStr.substring(0, 3);
            } else if (nanosStr.substring(6) === "000") {
                nanosStr = nanosStr.substring(0, 6);
            }
            text += "." + nanosStr;
            if (this.nanos < 0 && Number(this.seconds) == 0) {
                text = "-" + text;
            }
        }
        return text + "s";
    }
    static fromBinary(bytes, options) {
        return new Duration().fromBinary(bytes, options);
    }
    static fromJson(jsonValue, options) {
        return new Duration().fromJson(jsonValue, options);
    }
    static fromJsonString(jsonString, options) {
        return new Duration().fromJsonString(jsonString, options);
    }
    static equals(a, b) {
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto3$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["proto3"].util.equals(Duration, a, b);
    }
}
Duration.runtime = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto3$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["proto3"];
Duration.typeName = "google.protobuf.Duration";
Duration.fields = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f2e$pnpm$2f40$bufbuild$2b$protobuf$40$1$2e$10$2e$1$2f$node_modules$2f40$bufbuild$2f$protobuf$2f$dist$2f$esm$2f$proto3$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["proto3"].util.newFieldList(()=>[
        {
            no: 1,
            name: "seconds",
            kind: "scalar",
            T: 3 /* ScalarType.INT64 */ 
        },
        {
            no: 2,
            name: "nanos",
            kind: "scalar",
            T: 5 /* ScalarType.INT32 */ 
        }
    ]);
}}),

};

//# sourceMappingURL=51eda_%40bufbuild_protobuf_dist_esm_65605922._.js.map