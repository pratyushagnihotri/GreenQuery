import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GreenQuery Workbench",
  description: "Archived sustainable multi-modal query benchmarks",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className="h-full antialiased dark"
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
