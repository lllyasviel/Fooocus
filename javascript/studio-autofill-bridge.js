(function () {
    if (window.__fooocusStudioAutofillBridgeInstalled) {
        return;
    }
    window.__fooocusStudioAutofillBridgeInstalled = true;

    const allowedOrigins = new Set([
        "http://127.0.0.1:7872",
        "http://localhost:7872"
    ]);

    function findInput(elemId) {
        return document.querySelector(`#${elemId} textarea`)
            || document.querySelector(`textarea#${elemId}`)
            || document.querySelector(`#${elemId} input`)
            || document.querySelector(`input#${elemId}`);
    }

    function setNativeValue(element, value) {
        if (!element) {
            return false;
        }

        const prototype = element.tagName === "TEXTAREA"
            ? window.HTMLTextAreaElement.prototype
            : window.HTMLInputElement.prototype;
        const descriptor = Object.getOwnPropertyDescriptor(prototype, "value");

        if (descriptor && descriptor.set) {
            descriptor.set.call(element, value || "");
        } else {
            element.value = value || "";
        }

        element.dispatchEvent(new Event("input", { bubbles: true }));
        element.dispatchEvent(new Event("change", { bubbles: true }));
        return true;
    }

    function sendAutofillResult(event, result) {
        if (!event.source || !event.origin) {
            return;
        }
        event.source.postMessage({
            type: "fooocus-studio-autofill-result",
            promptFilled: result.promptFilled,
            negativeFilled: result.negativeFilled,
            workflow: result.workflow || "",
            fooocusArea: result.fooocusArea || "",
            setup_steps: result.setup_steps || "",
            next_shots: result.next_shots || "",
            updatedAt: result.updatedAt
        }, event.origin);
    }

    window.addEventListener("message", function (event) {
        if (!allowedOrigins.has(event.origin)) {
            return;
        }

        const payload = event.data || {};
        if (payload.type !== "fooocus-studio-autofill") {
            return;
        }

        const promptFilled = setNativeValue(findInput("positive_prompt"), payload.prompt);
        const negativeFilled = setNativeValue(findInput("negative_prompt"), payload.negative_prompt);

        window.__fooocusStudioLastAutofill = {
            promptFilled: promptFilled,
            negativeFilled: negativeFilled,
            workflow: payload.workflow || "",
            fooocusArea: payload.fooocus_area || "",
            setup_steps: payload.setup_steps || "",
            next_shots: payload.next_shots || "",
            updatedAt: Date.now()
        };

        sendAutofillResult(event, window.__fooocusStudioLastAutofill);

        if (!promptFilled || !negativeFilled) {
            console.warn("Fooocus Studio autofill could not find one or more target fields.", window.__fooocusStudioLastAutofill);
        }
    });
})();
