"use client";

import QRCode from "qrcode";
import { useEffect, useState } from "react";

// "Play on your phone": QR code to the Challenge. NEXT_PUBLIC_PLAY_URL (e.g. the public deployment) wins;
// otherwise this page's address, with localhost swapped for the machine's LAN address so phones on the same
// network can open it.
async function playUrl(): Promise<string> {
  if (process.env.NEXT_PUBLIC_PLAY_URL) return process.env.NEXT_PUBLIC_PLAY_URL;
  const { protocol, hostname, port, origin } = window.location;
  if (!["localhost", "127.0.0.1", "[::1]"].includes(hostname)) return `${origin}/?view=challenge`;
  const { lan } = await fetch("/lan-url").then((r) => r.json()).catch(() => ({ lan: null }));
  return `${protocol}//${lan ?? hostname}${port ? `:${port}` : ""}/?view=challenge`;
}

export function PlayQr() {
  const [code, setCode] = useState<{ url: string; svg: string } | null>(null);
  useEffect(() => {
    let alive = true;
    playUrl().then(async (url) => {
      const svg = await QRCode.toString(url, { type: "svg", margin: 1, errorCorrectionLevel: "M", color: { dark: "#1f2a24", light: "#ffffff" } });
      if (alive) setCode({ url, svg });
    });
    return () => { alive = false; };
  }, []);
  if (!code) return null;
  return (
    <a className="ch-qr" href={code.url} title={code.url}>
      <span className="ch-qr-code" dangerouslySetInnerHTML={{ __html: code.svg }} />
      <span className="ch-qr-text"><strong>Play on your phone</strong><small>{code.url.replace(/^https?:\/\//, "")}</small></span>
    </a>
  );
}
