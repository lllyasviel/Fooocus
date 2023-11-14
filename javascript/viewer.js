window.main_viewer_height = 512;

function refresh_grid() {
    let gridContainer = document.querySelector('#final_gallery .grid-container');
    let final_gallery = document.getElementById('final_gallery');

    if (gridContainer) if (final_gallery) {
        let rect = final_gallery.getBoundingClientRect();
        let cols = Math.ceil((rect.width - 16.0) / rect.height);
        if(cols < 2) cols = 2;
        gridContainer.style.setProperty('--grid-cols', cols);
    }
}

function refresh_grid_delayed() {
    refresh_grid();
    setTimeout(refresh_grid, 100);
    setTimeout(refresh_grid, 500);
    setTimeout(refresh_grid, 1000);
}

function resized() {
    let windowHeight = window.innerHeight - 260;
    let elements = document.getElementsByClassName('main_view');

    if (windowHeight > 745) windowHeight = 745;

    for (let i = 0; i < elements.length; i++) {
        elements[i].style.height = windowHeight + 'px';
    }

    window.main_viewer_height = windowHeight;

    refresh_grid();
}

function viewer_to_top(delay = 100) {
    setTimeout(() => window.scrollTo({top: 0, behavior: 'smooth'}), delay);
}

function viewer_to_bottom(delay = 100) {
    let element = document.getElementById('positive_prompt');
    let yPos = window.main_viewer_height;

    if (element) {
        yPos = element.getBoundingClientRect().top + window.scrollY;
    }

    setTimeout(() => window.scrollTo({top: yPos - 8, behavior: 'smooth'}), delay);
}

window.addEventListener('resize', (e) => {
    resized();
});

onUiLoaded(async () => {
    resized();
});
