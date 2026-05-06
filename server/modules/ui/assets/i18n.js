const translations = window.VI3DR_TRANSLATIONS || {};

const defaultLocale = "en";
let currentLocale = defaultLocale;

function resolveKey(locale, key) {
  return key.split(".").reduce((value, part) => value?.[part], translations[locale]);
}

function interpolate(template, params = {}) {
  return template.replace(/\{(\w+)\}/g, (_, key) => `${params[key] ?? ""}`);
}

function t(key, params = {}) {
  const localeValue = resolveKey(currentLocale, key);
  const fallbackValue = resolveKey(defaultLocale, key);
  const value = localeValue ?? fallbackValue ?? key;

  if (typeof value !== "string") {
    return key;
  }

  return interpolate(value, params);
}

function applyTranslations(root = document) {
  root.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = t(element.dataset.i18n);
  });
}

function setLocale(locale) {
  currentLocale = translations[locale] ? locale : defaultLocale;
  document.documentElement.lang = currentLocale;
  document.title = t("title");
  applyTranslations();
}

function getLocale() {
  return currentLocale;
}

window.VI3DR_I18N = {
  applyTranslations,
  getLocale,
  setLocale,
  supportedLocales: Object.keys(translations),
  t,
};
