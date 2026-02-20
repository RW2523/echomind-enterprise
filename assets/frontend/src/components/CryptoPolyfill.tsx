/*
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
*/
"use client";

import { useEffect } from "react";

/**
 * Polyfill for crypto.randomUUID when not available (runs once on mount, no script-in-html).
 */
export default function CryptoPolyfill() {
  useEffect(() => {
    if (typeof crypto !== "undefined" && !crypto.randomUUID) {
      (crypto as Crypto & { randomUUID?: () => string }).randomUUID = function randomUUID(): string {
        return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
          const r = (Math.random() * 16) | 0;
          const v = c === "x" ? r : (r & 0x3) | 0x8;
          return v.toString(16);
        });
      };
    }
  }, []);
  return null;
}
