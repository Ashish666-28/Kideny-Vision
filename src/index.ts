import { DurableObject } from "cloudflare:workers";

interface Env {
  KIDNEY_VISION: DurableObjectNamespace<KidneyVisionContainer>;
  GOOGLE_API_KEY?: string;
  GEMINI_MODEL?: string;
  FLASK_SECRET_KEY?: string;
}

const PORT = 8080;
const INACTIVITY_TIMEOUT_MS = 30 * 60 * 1000;
const HEALTHCHECK_ATTEMPTS = 100;
const HEALTHCHECK_DELAY_MS = 200;

export class KidneyVisionContainer extends DurableObject<Env> {
  private starting: Promise<void> | undefined;

  async fetch(request: Request): Promise<Response> {
    this.starting ??= this.startAndWaitForPort().finally(() => {
      this.starting = undefined;
    });
    await this.starting;

    const url = new URL(request.url);
    url.protocol = "http:";
    url.host = "container";
    const forwarded = new Request(url, request);
    forwarded.headers.delete("host");
    return this.ctx.container!.getTcpPort(PORT).fetch(forwarded);
  }

  private async startAndWaitForPort(): Promise<void> {
    const container = this.ctx.container;
    if (!container) {
      throw new Error("Cloudflare Container binding is unavailable.");
    }
    if (!container.running) {
      container.start({
        image: container.images.base,
        instance: "standard-1",
        enableInternet: true,
        env: {
          GOOGLE_API_KEY: this.env.GOOGLE_API_KEY ?? "",
          GEMINI_MODEL: this.env.GEMINI_MODEL ?? "gemini-flash-latest",
          FLASK_SECRET_KEY: this.env.FLASK_SECRET_KEY ?? "",
          HOST: "0.0.0.0",
          PORT: String(PORT),
        },
      });
    }

    await container.setInactivityTimeout(INACTIVITY_TIMEOUT_MS);
    const port = container.getTcpPort(PORT);
    let lastError: unknown;

    for (let attempt = 0; attempt < HEALTHCHECK_ATTEMPTS; attempt += 1) {
      try {
        const response = await port.fetch("http://container/health", {
          signal: AbortSignal.timeout(1000),
        });
        await response.body?.cancel();
        if (response.ok) return;
        lastError = new Error(`Health check returned ${response.status}.`);
      } catch (error) {
        lastError = error;
      }
      await scheduler.wait(HEALTHCHECK_DELAY_MS);
    }

    throw new Error("KidneyVision did not become ready on port 8080.", {
      cause: lastError,
    });
  }
}

export default {
  fetch(request: Request, env: Env): Promise<Response> {
    return env.KIDNEY_VISION.getByName("primary").fetch(request);
  },
} satisfies ExportedHandler<Env>;
