# Astro learning frontend

Static, read-only teaching interface for the Throughput Agent research repo. No backend, GPU jobs or cloud services.

## Local use

```sh
cd web
npm install
npm run dev
```

For static build and type checks:

```sh
npm run build
```

Open the local URL displayed by Astro (normally http://localhost:4321).

The experiment page reads the committed task YAML files directly at build time. The learning calculator is conceptual, not a performance prediction. There are no measured results yet. The source repo remains the authoritative record.

Dependencies are managed with npm under web/. Python uses uv at repository root.
