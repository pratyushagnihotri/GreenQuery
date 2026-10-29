import { networkInterfaces } from "node:os";

// First LAN IPv4 address of the machine serving the dashboard, so the Challenge QR code opened on localhost
// still points phones at an address they can reach.
export function GET() {
  const nets = Object.values(networkInterfaces()).flat()
    .filter((net) => net && net.family === "IPv4" && !net.internal).map((net) => net!.address);
  const lan = nets.find((ip) => /^(192\.168|10\.|172\.(1[6-9]|2\d|3[01])\.)/.test(ip)) ?? nets[0] ?? null;
  return Response.json({ lan });
}
