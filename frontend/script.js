const form =
    document.getElementById("searchForm");

const input =
    document.getElementById("searchInput");

const logoButton =
    document.getElementById("logoButton");

const inputWrapper =
    document.getElementById("inputWrapper");

const validationMessage =
    document.getElementById("validationMessage");

const clearButton =
    document.getElementById("clearButton");

const loadingState =
    document.getElementById("loadingState");

const resultsState =
    document.getElementById("resultsState");

const noResultsState =
    document.getElementById("noResultsState");

const drawer =
    document.getElementById("drawer");

const drawerOverlay =
    document.getElementById("drawerOverlay");

const closeDrawerButton =
    document.getElementById("closeDrawer");

const externalSourceButton =
    document.getElementById("externalSourceButton");

const drawerTitle =
    document.getElementById("drawerTitle");

const drawerOrganisation =
    document.getElementById("drawerOrganisation");

const drawerType =
    document.getElementById("drawerType");

const drawerNumber =
    document.getElementById("drawerNumber");

const drawerFragment =
    document.getElementById("drawerFragment");


let searchTimer = null;


/* ==============================
   DEMO SOURCES
============================== */

const sources = {

    first: {

        title:
            "Положение Банка России № 762-П",

        organisation:
            "Банк России",

        type:
            "Нормативный документ",

        number:
            "№ 762-П",

        fragment:
            "Кредитная организация при проведении операций руководствуется установленными законодательством требованиями и внутренними процедурами контроля."

    },


    second: {

        title:
            "Федеральный закон № 115-ФЗ",

        organisation:
            "Российская Федерация",

        type:
            "Федеральный закон",

        number:
            "№ 115-ФЗ",

        fragment:
            "Кредитная организация может запросить сведения и документы в случаях, предусмотренных требованиями законодательства, регулирующего проведение финансовых операций."

    }

};


/* ==============================
   STATES
============================== */

function hideStates() {

    loadingState.classList.add("hidden");

    resultsState.classList.add("hidden");

    noResultsState.classList.add("hidden");

}


function clearValidation() {

    inputWrapper.classList.remove("error");

    validationMessage.classList.remove("visible");

}


function showValidation() {

    inputWrapper.classList.add("error");

    validationMessage.classList.add("visible");

}


function updateClearButton() {

    const hasText =
        input.value.trim().length > 0;


    clearButton.classList.toggle(
        "visible",
        hasText
    );

}


/* ==============================
   DRAWER
============================== */

function openDrawer(sourceKey) {

    const source =
        sources[sourceKey];


    if (!source) {
        return;
    }


    drawerTitle.textContent =
        source.title;

    drawerOrganisation.textContent =
        source.organisation;

    drawerType.textContent =
        source.type;

    drawerNumber.textContent =
        source.number;

    drawerFragment.textContent =
        source.fragment;


    drawer.classList.add("open");

    drawerOverlay.classList.remove("hidden");

    document.body.style.overflow =
        "hidden";

}


function closeDrawer() {

    drawer.classList.remove("open");

    drawerOverlay.classList.add("hidden");

    document.body.style.overflow =
        "";

}


/* ==============================
   HOME
============================== */

function resetToInitial() {

    if (searchTimer) {

        clearTimeout(searchTimer);

        searchTimer = null;

    }


    hideStates();

    closeDrawer();

    clearValidation();


    document.body.classList.remove(
        "search-active"
    );


    input.value = "";

    updateClearButton();


    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });


    setTimeout(
        () => {
            input.focus();
        },
        150
    );

}


/* ==============================
   INPUT EVENTS
============================== */

input.addEventListener(
    "input",
    () => {

        updateClearButton();


        if (input.value.trim()) {
            clearValidation();
        }

    }
);


clearButton.addEventListener(
    "click",
    () => {

        input.value = "";

        updateClearButton();

        clearValidation();

        input.focus();

    }
);


/* ==============================
   EXAMPLE QUERIES
============================== */

document
    .querySelectorAll("[data-query]")
    .forEach((button) => {

        button.addEventListener(
            "click",
            () => {

                input.value =
                    button.dataset.query;

                updateClearButton();

                clearValidation();

                input.focus();

            }
        );

    });


/* ==============================
   SEARCH
============================== */

form.addEventListener(
    "submit",
    (event) => {

        event.preventDefault();


        const query =
            input.value.trim();


        hideStates();

        clearValidation();

        closeDrawer();


        /* пустой запрос */

        if (!query) {

            document.body.classList.remove(
                "search-active"
            );

            showValidation();

            input.focus();

            return;

        }


        /* переходим в compact mode */

        document.body.classList.add(
            "search-active"
        );


        /* loading */

        loadingState.classList.remove(
            "hidden"
        );


        if (searchTimer) {

            clearTimeout(searchTimer);

        }


        searchTimer =
            setTimeout(
                () => {

                    loadingState.classList.add(
                        "hidden"
                    );


                    const normalized =
                        query.toLowerCase();


                    /* demo-сценарий "ничего не найдено" */

                    const noResults =
                        normalized.includes("квант") ||
                        normalized.includes("марсиан") ||
                        normalized.includes("нейтрино") ||
                        normalized.includes("астроном");


                    if (noResults) {

                        noResultsState.classList.remove(
                            "hidden"
                        );

                    } else {

                        resultsState.classList.remove(
                            "hidden"
                        );

                    }

                },
                950
            );

    }
);


/* ==============================
   RESULT SOURCE BUTTONS
============================== */

document
    .querySelectorAll(".source-button")
    .forEach((button) => {

        button.addEventListener(
            "click",
            () => {

                openDrawer(
                    button.dataset.source
                );

            }
        );

    });


/* ==============================
   ANSWER SOURCE NUMBERS
============================== */

document
    .querySelectorAll(
        "[data-answer-source]"
    )
    .forEach((button) => {

        button.addEventListener(
            "click",
            () => {

                openDrawer(
                    button.dataset.answerSource
                );

            }
        );

    });


/* ==============================
   CLOSE DRAWER
============================== */

closeDrawerButton.addEventListener(
    "click",
    closeDrawer
);


drawerOverlay.addEventListener(
    "click",
    closeDrawer
);


document.addEventListener(
    "keydown",
    (event) => {

        if (event.key === "Escape") {

            closeDrawer();

        }

    }
);


/* ==============================
   EXTERNAL SOURCE
============================== */

externalSourceButton.addEventListener(
    "click",
    () => {

        alert(
            "В демонстрационной версии переход к внешнему нормативному источнику пока не подключён."
        );

    }
);


/* ==============================
   LOGO = HOME
============================== */

logoButton.addEventListener(
    "click",
    resetToInitial
);


/* ==============================
   INIT
============================== */

updateClearButton();
