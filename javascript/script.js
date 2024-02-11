// based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.6.0/script.js
function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const elem = elems.length == 0 ? document : elems[0];

    if (elem !== document) {
        elem.getElementById = function(id) {
            return document.getElementById(id);
        };
    }
    return elem.shadowRoot ? elem.shadowRoot : elem;
}

/**
 * Get the currently selected top-level UI tab button (e.g. the button that says "Extras").
 */
function get_uiCurrentTab() {
    return gradioApp().querySelector('#tabs > .tab-nav > button.selected');
}

/**
 * Get the first currently visible top-level UI tab content (e.g. the div hosting the "txt2img" UI).
 */
function get_uiCurrentTabContent() {
    return gradioApp().querySelector('#tabs > .tabitem[id^=tab_]:not([style*="display: none"])');
}

var uiUpdateCallbacks = [];
var uiAfterUpdateCallbacks = [];
var uiLoadedCallbacks = [];
var uiTabChangeCallbacks = [];
var optionsChangedCallbacks = [];
var uiAfterUpdateTimeout = null;
var uiCurrentTab = null;

/**
 * Register callback to be called at each UI update.
 * The callback receives an array of MutationRecords as an argument.
 */
function onUiUpdate(callback) {
    uiUpdateCallbacks.push(callback);
}

/**
 * Register callback to be called soon after UI updates.
 * The callback receives no arguments.
 *
 * This is preferred over `onUiUpdate` if you don't need
 * access to the MutationRecords, as your function will
 * not be called quite as often.
 */
function onAfterUiUpdate(callback) {
    uiAfterUpdateCallbacks.push(callback);
}

/**
 * Register callback to be called when the UI is loaded.
 * The callback receives no arguments.
 */
function onUiLoaded(callback) {
    uiLoadedCallbacks.push(callback);
}

/**
 * Register callback to be called when the UI tab is changed.
 * The callback receives no arguments.
 */
function onUiTabChange(callback) {
    uiTabChangeCallbacks.push(callback);
}

/**
 * Register callback to be called when the options are changed.
 * The callback receives no arguments.
 * @param callback
 */
function onOptionsChanged(callback) {
    optionsChangedCallbacks.push(callback);
}

function executeCallbacks(queue, arg) {
    for (const callback of queue) {
        try {
            callback(arg);
        } catch (e) {
            console.error("error running callback", callback, ":", e);
        }
    }
}

/**
 * Schedule the execution of the callbacks registered with onAfterUiUpdate.
 * The callbacks are executed after a short while, unless another call to this function
 * is made before that time. IOW, the callbacks are executed only once, even
 * when there are multiple mutations observed.
 */
function scheduleAfterUiUpdateCallbacks() {
    clearTimeout(uiAfterUpdateTimeout);
    uiAfterUpdateTimeout = setTimeout(function() {
        executeCallbacks(uiAfterUpdateCallbacks);
    }, 200);
}

var executedOnLoaded = false;

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m) {
        if (!executedOnLoaded && gradioApp().querySelector('#generate_button')) {
            executedOnLoaded = true;
            executeCallbacks(uiLoadedCallbacks);
        }

        executeCallbacks(uiUpdateCallbacks, m);
        scheduleAfterUiUpdateCallbacks();
        const newTab = get_uiCurrentTab();
        if (newTab && (newTab !== uiCurrentTab)) {
            uiCurrentTab = newTab;
            executeCallbacks(uiTabChangeCallbacks);
        }
    });
    mutationObserver.observe(gradioApp(), {childList: true, subtree: true});
    initStylePreviewOverlay();
    initModelPreviewOverlay();
});

/**
 * Add a ctrl+enter as a shortcut to start a generation
 */
document.addEventListener('keydown', function(e) {
    const isModifierKey = (e.metaKey || e.ctrlKey || e.altKey);
    const isEnterKey = (e.key == "Enter" || e.keyCode == 13);

    if(isModifierKey && isEnterKey) {
        const generateButton = gradioApp().querySelector('button:not(.hidden)[id=generate_button]');
        if (generateButton) {
            generateButton.click();
            e.preventDefault();
            return;
        }

        const stopButton = gradioApp().querySelector('button:not(.hidden)[id=stop_button]')
        if(stopButton) {
            stopButton.click();
            e.preventDefault();
            return;
        }
    }
});

// Utility functions
function formatImagePath(name, templateImagePath, replacedValue = "fooocus_v2") {
    return templateImagePath.replace(replacedValue, name.toLowerCase().replaceAll(" ", "_")).replaceAll("\\", "\\\\");
}

function createOverlay(id) {
    const overlay = document.createElement('div');
    overlay.id = id;
    document.body.appendChild(overlay);
    return overlay;
}

function setImageBackground(overlay, url) {
    unsetOverlayAsTooltip(overlay)
    overlay.style.backgroundImage = `url("${url}")`;
}

function setOverlayAsTooltip(overlay, altText) {
    // Set the text content and any dynamic styles
    overlay.textContent = altText;
    overlay.style.width = 'fit-content';
    overlay.style.height = 'fit-content';
    // Note: Other styles are already set via CSS
}

function unsetOverlayAsTooltip(overlay) {
    // Clear the text content and reset any dynamic styles
    overlay.textContent = '';
    overlay.style.width = '128px';
    overlay.style.height = '128px';
    // Note: Other styles are managed via CSS
}

function handleMouseMove(overlay) {
    return function(e) {
        if (overlay.style.opacity !== "1") return;
        overlay.style.left = `${e.clientX}px`;
        overlay.style.top = `${e.clientY}px`;
        overlay.className = e.clientY > window.innerHeight / 2 ? "lower-half" : "upper-half";
    };
}

// Image path retrieval for models
const getModelImagePath = selectedItemText => {
    selectedItemText = selectedItemText.replace("✓\n", "")

    let imagePath = null;

    if (previewsCheckpoint)
        imagePath = previewsCheckpoint[selectedItemText]
    
    if (previewsLora && !imagePath)
        imagePath = previewsLora[selectedItemText]
    
    return imagePath;
};

// Mouse over handlers for different overlays
function handleMouseOverModelPreviewOverlay(overlay, elementSelector, templateImagePath) {
    return function(e) {
        const targetElement = e.target.closest(elementSelector);
        if (!targetElement) return;

        targetElement.removeEventListener("mouseout", onMouseLeave);
        targetElement.addEventListener("mouseout", onMouseLeave);

        overlay.style.opacity = "1";
        const selectedItemText = targetElement.innerText;
        if (selectedItemText) {
            let imagePath = getModelImagePath(selectedItemText);
            if (imagePath) {
                imagePath = formatImagePath(imagePath, templateImagePath, "sdxl_styles/samples/fooocus_v2.jpg");
                setImageBackground(overlay, imagePath);
            } else {
                setOverlayAsTooltip(overlay, selectedItemText);
            }
        }

        function onMouseLeave() {
            overlay.style.opacity = "0";
            overlay.style.backgroundImage = "";
            targetElement.removeEventListener("mouseout", onMouseLeave);
        }
    };
}

function handleMouseOverStylePreviewOverlay(overlay, elementSelector, templateImagePath) {
    return function(e) {
        const label = e.target.closest(elementSelector);
        if (!label) return;

        label.removeEventListener("mouseout", onMouseLeave);
        label.addEventListener("mouseout", onMouseLeave);

        overlay.style.opacity = "1";

        const originalText = label.querySelector("span").getAttribute("data-original-text");
        let name = originalText || label.querySelector("span").textContent;
        let imagePath = formatImagePath(name, templateImagePath);

        overlay.style.backgroundImage = `url("${imagePath}")`;

        function onMouseLeave() {
            overlay.style.opacity = "0";
            overlay.style.backgroundImage = "";
            label.removeEventListener("mouseout", onMouseLeave);
        }
    };
}

// Initialization functions for different overlays
function initModelPreviewOverlay() {
    const templateImagePath = document.querySelector("meta[name='samples-path']").getAttribute("content");
    const modelOverlay = createOverlay('modelPreviewOverlay');

    document.addEventListener('mouseover', handleMouseOverModelPreviewOverlay(
        modelOverlay, 
        '.model_selections .item', 
        templateImagePath
    ));

    document.addEventListener('mousemove', handleMouseMove(modelOverlay));
}

function initStylePreviewOverlay() {
    const templateImagePath = document.querySelector("meta[name='samples-path']").getAttribute("content");
    const styleOverlay = createOverlay('stylePreviewOverlay');

    document.addEventListener('mouseover', handleMouseOverStylePreviewOverlay(
        styleOverlay, 
        '.style_selections label', 
        templateImagePath
    ));

    document.addEventListener('mousemove', handleMouseMove(styleOverlay));
}

/**
 * checks that a UI element is not in another hidden element or tab content
 */
function uiElementIsVisible(el) {
    if (el === document) {
        return true;
    }

    const computedStyle = getComputedStyle(el);
    const isVisible = computedStyle.display !== 'none';

    if (!isVisible) return false;
    return uiElementIsVisible(el.parentNode);
}

function uiElementInSight(el) {
    const clRect = el.getBoundingClientRect();
    const windowHeight = window.innerHeight;
    const isOnScreen = clRect.bottom > 0 && clRect.top < windowHeight;

    return isOnScreen;
}

function playNotification() {
    gradioApp().querySelector('#audio_notification audio')?.play();
}

function set_theme(theme) {
    var gradioURL = window.location.href;
    if (!gradioURL.includes('?__theme=')) {
        window.location.replace(gradioURL + '?__theme=' + theme);
    }
}
