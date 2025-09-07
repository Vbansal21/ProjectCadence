export const config = {
    env: process.env.NODE_ENV ?? "development",
    port: Number(process.env.PORT ?? 3000),
    apiKey: process.env.API_KEY ?? "",
    cors: { origin: true, credentials: true },
    rate: { windowMs: 60_000, max: 120 }
  };
