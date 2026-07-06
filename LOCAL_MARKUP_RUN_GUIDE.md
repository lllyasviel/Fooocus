# Local Markup Run Guide

This branch adds two simple local launchers.

## Run Fooocus

Windows:

```bat
RUN_FOOOCUS_LOCAL.bat
```

Mac/Linux:

```bash
bash RUN_FOOOCUS_LOCAL.sh
```

Fooocus runs on the normal local port.

## Run the Markup Assistant

Windows:

```bat
RUN_MARKUP_ASSISTANT.bat
```

Mac/Linux:

```bash
bash RUN_MARKUP_ASSISTANT.sh
```

The Markup Assistant opens at:

```txt
http://127.0.0.1:7871
```

## Workflow

1. Open Fooocus.
2. Open the Markup Assistant.
3. Type an edit like: `make the shirt black and remove the trash`.
4. Press **Understand Edit**.
5. Copy the generated prompt into Fooocus Inpaint.
6. Draw your mask and generate.

This is local-only. No RunPod or remote GPU changes are included.
