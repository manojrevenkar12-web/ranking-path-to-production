import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";

import styles from "./Shell.module.css";

const NAV = [
  { to: "/", label: "Path", end: true },
  { to: "/runs", label: "Pipeline runs" },
  { to: "/models", label: "Model versions" },
  { to: "/routing", label: "Routing" },
  { to: "/standards", label: "Standards" },
  { to: "/evidence", label: "Evidence" },
];

export function Shell({ children }: { children: ReactNode }) {
  return (
    <div className={styles.shell}>
      <a className={styles.skip} href="#main">
        Skip to content
      </a>
      <header className={styles.side}>
        <div className={styles.brand}>
          <svg viewBox="0 0 40 16" className={styles.mark} aria-hidden>
            <path d="M2 12 H12 L20 4 L28 12 H38" />
          </svg>
          <span>
            Buyer models
            <small>path to production</small>
          </span>
        </div>
        <nav aria-label="Sections">
          <ul className={styles.nav}>
            {NAV.map((n) => (
              <li key={n.to}>
                <NavLink
                  to={n.to}
                  end={n.end}
                  className={({ isActive }) => (isActive ? styles.active : undefined)}
                >
                  {n.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </header>
      <main id="main" className={styles.main} tabIndex={-1}>
        {children}
      </main>
    </div>
  );
}
