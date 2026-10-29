import { networkInterfaces } from "node:os";
import type { NextConfig } from "next";

// IPv4 addresses of this machine, so phones on the same network can open the dev server (QR code on the
// Challenge page) without Next blocking them as a cross-origin dev client.
const lanHosts = Object.values(networkInterfaces()).flat()
  .filter((net) => net && net.family === "IPv4" && !net.internal).map((net) => net!.address);
const apiTarget = (process.env.NEXT_PUBLIC_GREENQUERY_API ?? "http://localhost:8000").replace(/\/$/, "");

const nextConfig: NextConfig = {
  output: "standalone",
  devIndicators: false, // no dev badge over the sidebar (demos, screenshots); errors still show
  allowedDevOrigins: lanHosts,
  turbopack: {
    root: __dirname,
  },
  // Pages opened from another device can't reach a localhost API; they call /gq-api, which this server
  // forwards to the GreenQuery service (see API_BASE in src/lib/api.ts).
  async rewrites() {
    return [{ source: "/gq-api/:path*", destination: `${apiTarget}/:path*` }];
  },
};

export default nextConfig;
