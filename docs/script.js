(() => {
  const body = document.body;
  const toggle = document.querySelector("[data-nav-toggle]");
  const panel = document.querySelector("[data-nav-panel]");

  const closeNav = () => {
    body.classList.remove("nav-open");
    if (toggle) {
      toggle.setAttribute("aria-expanded", "false");
    }
  };

  if (toggle && panel) {
    toggle.addEventListener("click", () => {
      const isOpen = body.classList.toggle("nav-open");
      toggle.setAttribute("aria-expanded", String(isOpen));
    });

    panel.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", closeNav);
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        closeNav();
      }
    });

    window.matchMedia("(min-width: 961px)").addEventListener("change", (event) => {
      if (event.matches) {
        closeNav();
      }
    });
  }

  // Desktop dropdowns
  document.querySelectorAll(".nav__dropdown-toggle").forEach((button) => {
    const dropdown = button.closest(".nav__dropdown");
    const menu = dropdown.querySelector(".nav__dropdown-menu");

    button.addEventListener("click", (e) => {
      e.stopPropagation();
      const isActive = dropdown.classList.toggle("active");
      button.setAttribute("aria-expanded", String(isActive));
    });

    // Close dropdown when clicking outside
    document.addEventListener("click", (e) => {
      if (!dropdown.contains(e.target)) {
        dropdown.classList.remove("active");
        button.setAttribute("aria-expanded", "false");
      }
    });

    // Close dropdown on escape
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && dropdown.classList.contains("active")) {
        dropdown.classList.remove("active");
        button.setAttribute("aria-expanded", "false");
      }
    });
  });

  // Mobile dropdowns
  document.querySelectorAll(".nav__mobile-dropdown-toggle").forEach((button) => {
    const dropdown = button.closest(".nav__mobile-dropdown");

    button.addEventListener("click", () => {
      const isActive = dropdown.classList.toggle("active");
      button.setAttribute("aria-expanded", String(isActive));
    });
  });

  const yearEl = document.getElementById("year");
  if (yearEl) {
    yearEl.textContent = new Date().getFullYear().toString();
  }

  // Carousel functionality
  const carouselTrack = document.getElementById("carouselTrack");
  const carouselPrev = document.getElementById("carouselPrev");
  const carouselNext = document.getElementById("carouselNext");
  const carouselDots = document.querySelectorAll(".carousel-dot");

  if (carouselTrack && carouselDots.length > 0) {
    let currentSlide = 0;
    const totalSlides = carouselDots.length;
    let autoPlayInterval = null;

    function updateCarousel() {
      const translateX = -currentSlide * 100;
      carouselTrack.style.transform = `translate3d(${translateX}%, 0, 0)`;
      
      carouselDots.forEach((dot, index) => {
        if (index === currentSlide) {
          dot.classList.add("active");
        } else {
          dot.classList.remove("active");
        }
      });

      // Update text content below carousel
      const activeSlide = carouselTrack.querySelector(`[data-slide-index="${currentSlide}"]`);
      const titleEl = document.getElementById("carouselTitle");
      const descriptionEl = document.getElementById("carouselDescription");
      
      if (activeSlide && titleEl && descriptionEl) {
        const title = activeSlide.getAttribute("data-title");
        const description = activeSlide.getAttribute("data-description");
        
        if (title) {
          titleEl.style.opacity = "0";
          setTimeout(() => {
            titleEl.textContent = title;
            titleEl.style.opacity = "1";
          }, 150);
        }
        
        if (description) {
          descriptionEl.style.opacity = "0";
          setTimeout(() => {
            descriptionEl.textContent = description;
            descriptionEl.style.opacity = "1";
          }, 150);
        }
      }
    }

    function nextSlide() {
      currentSlide = (currentSlide + 1) % totalSlides;
      updateCarousel();
    }

    function prevSlide() {
      currentSlide = (currentSlide - 1 + totalSlides) % totalSlides;
      updateCarousel();
    }

    function goToSlide(index) {
      currentSlide = index;
      updateCarousel();
    }

    function startAutoPlay() {
      stopAutoPlay();
      autoPlayInterval = setInterval(nextSlide, 5000);
    }

    function stopAutoPlay() {
      if (autoPlayInterval) {
        clearInterval(autoPlayInterval);
        autoPlayInterval = null;
      }
    }

    updateCarousel();
    startAutoPlay();
    
    if (carouselNext) {
      carouselNext.addEventListener("click", (e) => {
        e.preventDefault();
        nextSlide();
        stopAutoPlay();
        startAutoPlay();
      });
    }
    
    if (carouselPrev) {
      carouselPrev.addEventListener("click", (e) => {
        e.preventDefault();
        prevSlide();
        stopAutoPlay();
        startAutoPlay();
      });
    }
    
    carouselDots.forEach((dot, index) => {
      dot.addEventListener("click", (e) => {
        e.preventDefault();
        goToSlide(index);
        stopAutoPlay();
        startAutoPlay();
      });
    });
    
    const carouselContainer = document.querySelector(".carousel-container");
    if (carouselContainer) {
      carouselContainer.addEventListener("mouseenter", stopAutoPlay);
      carouselContainer.addEventListener("mouseleave", startAutoPlay);
    }
    
    const carousel = document.querySelector(".hero-carousel");
    if (carousel) {
      const handleKeydown = (e) => {
        if (document.activeElement === document.body || carousel.contains(document.activeElement)) {
          if (e.key === "ArrowLeft") {
            e.preventDefault();
            prevSlide();
            stopAutoPlay();
            startAutoPlay();
          } else if (e.key === "ArrowRight") {
            e.preventDefault();
            nextSlide();
            stopAutoPlay();
            startAutoPlay();
          }
        }
      };
      
      document.addEventListener("keydown", handleKeydown);
    }
  }

  // Media carousels in sections
  document.querySelectorAll("[data-carousel]").forEach((carousel) => {
    const track = carousel.querySelector("[data-carousel-track]");
    const slides = carousel.querySelectorAll("[data-carousel-slide]");
    const prev = carousel.querySelector("[data-carousel-prev]");
    const next = carousel.querySelector("[data-carousel-next]");
    const dots = carousel.querySelectorAll("[data-carousel-dot]");

    if (!track || slides.length === 0) {
      return;
    }

    if (slides.length <= 1) {
      if (prev) {
        prev.hidden = true;
      }
      if (next) {
        next.hidden = true;
      }
      const dotsContainer = carousel.querySelector("[data-carousel-dots]");
      if (dotsContainer) {
        dotsContainer.hidden = true;
      }
      return;
    }

    let currentIndex = 0;

    const update = () => {
      track.style.transform = `translate3d(-${currentIndex * 100}%, 0, 0)`;
      dots.forEach((dot, index) => {
        const isActive = index === currentIndex;
        dot.classList.toggle("is-active", isActive);
        dot.setAttribute("aria-selected", String(isActive));
      });
    };

    const goTo = (index) => {
      currentIndex = (index + slides.length) % slides.length;
      update();
    };

    if (prev) {
      prev.addEventListener("click", () => goTo(currentIndex - 1));
    }

    if (next) {
      next.addEventListener("click", () => goTo(currentIndex + 1));
    }
    dots.forEach((dot) => {
      const targetIndex = Number(dot.dataset.carouselDot) || 0;
      dot.addEventListener("click", () => goTo(targetIndex));
    });

    update();
  });

  // Docs search (only active on docs layout)
  const docsSearchInput = document.querySelector("[data-docs-search-input]");
  const docsSearchResults = document.querySelector("[data-docs-search-results]");
  const docsContent = document.querySelector(".docs-content");

  if (docsSearchInput && docsSearchResults) {
    let docsIndex = [];
    let indexLoaded = false;
    let indexLoading = false;

    const hideResults = () => {
      docsSearchResults.classList.remove("is-visible");
      docsSearchResults.innerHTML = "";
    };

    const renderResults = (results, query) => {
      docsSearchResults.innerHTML = "";

      if (!query || query.length < 2) {
        hideResults();
        return;
      }

      if (!results || results.length === 0) {
        docsSearchResults.innerHTML =
          '<div class="docs-search__empty">No matches found.</div>';
        docsSearchResults.classList.add("is-visible");
        return;
      }

      const fragment = document.createDocumentFragment();

      results.forEach((page) => {
        const item = document.createElement("a");
        item.className = "docs-search__item";
        item.href = page.url;

        const titleEl = document.createElement("div");
        titleEl.className = "docs-search__item-title";
        titleEl.textContent = page.title;

        const snippetEl = document.createElement("div");
        snippetEl.className = "docs-search__item-snippet";

        const content = page.content || "";
        const lower = content.toLowerCase();
        const idx = lower.indexOf(query.toLowerCase());

        if (idx !== -1) {
          const start = Math.max(0, idx - 40);
          const end = Math.min(content.length, idx + query.length + 60);
          let snippet = content.slice(start, end).trim();
          if (start > 0) snippet = "…" + snippet;
          if (end < content.length) snippet = snippet + "…";
          snippetEl.textContent = snippet;
        } else {
          snippetEl.textContent = content.slice(0, 120).trim() + "…";
        }

        item.appendChild(titleEl);
        item.appendChild(snippetEl);
        fragment.appendChild(item);
      });

      docsSearchResults.appendChild(fragment);
      docsSearchResults.classList.add("is-visible");
    };

    const loadIndex = async () => {
      if (indexLoaded || indexLoading) return;
      indexLoading = true;
      try {
        const response = await fetch("/search.json", { cache: "no-store" });
        if (response.ok) {
          docsIndex = await response.json();
          indexLoaded = true;
        }
      } catch (e) {
        // eslint-disable-next-line no-console
        console.error("Failed to load docs search index", e);
      } finally {
        indexLoading = false;
      }
    };

    const performSearch = async (query) => {
      const q = query.trim().toLowerCase();

      if (!q || q.length < 2) {
        hideResults();
        return;
      }

      await loadIndex();

      if (!docsIndex || docsIndex.length === 0) {
        hideResults();
        return;
      }

      const matches = docsIndex
        .map((page) => {
          const title = (page.title || "").toString();
          const content = (page.content || "").toString();
          const haystack = (title + " " + content).toLowerCase();
          const score = haystack.indexOf(q);
          return score === -1 ? null : { page, score };
        })
        .filter(Boolean)
        .sort((a, b) => a.score - b.score)
        .slice(0, 10)
        .map((entry) => entry.page);

      renderResults(matches, q);
    };

    let searchDebounce = null;

    docsSearchInput.addEventListener("input", (event) => {
      const value = event.target.value || "";
      if (searchDebounce) {
        window.clearTimeout(searchDebounce);
      }
      searchDebounce = window.setTimeout(() => {
        performSearch(value);
      }, 120);
    });

    docsSearchInput.addEventListener("focus", () => {
      if (docsSearchInput.value && docsSearchInput.value.length >= 2) {
        performSearch(docsSearchInput.value);
      }
    });

    document.addEventListener("click", (event) => {
      if (
        !docsSearchResults.contains(event.target) &&
        event.target !== docsSearchInput
      ) {
        hideResults();
      }
    });

    docsSearchInput.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        docsSearchInput.blur();
        hideResults();
      }
    });
  }

  // Docs-only enhancements: code copy buttons + heading anchor links
  if (docsContent) {
    // Add copy buttons to code blocks (handles both Rouge-highlighted and plain code)
    const addCopyButtons = () => {
      // Find all code elements, but process them intelligently to avoid duplicates
      const allCodeElements = docsContent.querySelectorAll("pre > code");
      
      allCodeElements.forEach((codeEl) => {
        // Skip if this code element is already inside a .docs-code-block wrapper
        if (codeEl.closest(".docs-code-block")) return;
        
        // Check if this is inside a Rouge .highlight block
        const highlightEl = codeEl.closest(".highlight");
        const containerEl = highlightEl || codeEl.parentElement; // Use .highlight if present, otherwise use pre
        const pre = codeEl.parentElement;
        
        // Skip if container is already enhanced
        if (containerEl.dataset.copyEnhanced === "true") return;
        
        containerEl.dataset.copyEnhanced = "true";
        const wrapper = document.createElement("div");
        wrapper.className = "docs-code-block";

        containerEl.parentNode.insertBefore(wrapper, containerEl);
        wrapper.appendChild(containerEl);

        const button = document.createElement("button");
        button.type = "button";
        button.className = "docs-code-copy-btn";
        button.setAttribute("aria-label", "Copy code to clipboard");
        button.textContent = "Copy";

        button.addEventListener("click", async () => {
          const text = codeEl.innerText || codeEl.textContent || "";
          try {
            await navigator.clipboard.writeText(text);
            button.classList.add("copied");
            button.textContent = "Copied";
            window.setTimeout(() => {
              button.classList.remove("copied");
              button.textContent = "Copy";
            }, 1500);
          } catch (e) {
            // eslint-disable-next-line no-console
            console.error("Failed to copy code", e);
          }
        });

        wrapper.appendChild(button);
      });
    };

    // Add anchor links to headings with IDs
    const addHeadingAnchors = () => {
      const headings = docsContent.querySelectorAll("h1[id], h2[id], h3[id], h4[id]");
      headings.forEach((heading) => {
        if (heading.querySelector(".docs-heading-anchor")) return;
        const id = heading.id;
        if (!id) return;

        const anchor = document.createElement("a");
        anchor.className = "docs-heading-anchor";
        anchor.href = `#${id}`;
        anchor.setAttribute("aria-label", "Link to this section");

        heading.insertBefore(anchor, heading.firstChild);
      });
    };

    // Wrap tables in scrollable containers for mobile
    const wrapTables = () => {
      const tables = docsContent.querySelectorAll("table");
      tables.forEach((table) => {
        // Skip if already wrapped
        if (table.parentElement.classList.contains("docs-table-wrapper")) return;
        
        const wrapper = document.createElement("div");
        wrapper.className = "docs-table-wrapper";
        
        table.parentNode.insertBefore(wrapper, table);
        wrapper.appendChild(table);
      });
    };

    addCopyButtons();
    addHeadingAnchors();
    wrapTables();
  }
})();

