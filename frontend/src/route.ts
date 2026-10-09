import { useEffect, useState } from "react";

export type Route = "home" | "predict";

const read = (): Route => (window.location.hash.startsWith("#/predict") ? "predict" : "home");

/** Hash routing: no router dependency, works on any static host and under the FastAPI SPA fallback. */
export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(read);
  useEffect(() => {
    const onChange = () => setRoute(read());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

export const go = (route: Route) => {
  window.location.hash = route === "home" ? "#/" : "#/predict";
};
