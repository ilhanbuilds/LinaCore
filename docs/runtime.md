# Runtime

Lifecycle moves through created, starting, ready, stopping, stopped or failed with locked, validated transitions. Startup resolves and loads registered providers; failure moves to failed. Shutdown unloads the active provider and clears selection.

Inference is rejected until ready. Streaming emits ordered started/token/terminal events. Cancellation is an event-based foundation for provider cooperation. Health is degraded before startup and delegates to the active provider when ready. Registries reject duplicates and missing IDs. The runtime is independent from HTTP, named pipes and IPC.

