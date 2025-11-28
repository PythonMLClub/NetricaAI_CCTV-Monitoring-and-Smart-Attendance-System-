document.addEventListener('DOMContentLoaded', async () => {
  try {
    // Find the topbar element (either 'topbar', 'topbar-placeholder', or 'topbar-container')
    const topbarElement = document.getElementById('topbar') ||
                         document.getElementById('topbar-placeholder') ||
                         document.getElementById('topbar-container');
    
    // Find the menus element
    const menusElement = document.getElementById('menus');
    
    // Check if elements exist
    if (!topbarElement) {
      console.error('Topbar element not found in the DOM');
      return;
    }
    if (!menusElement) {
      console.error('Menus element not found in the DOM');
      return;
    }

    // Check if topbar content is already loaded to avoid redundant fetches
    if (topbarElement.innerHTML.trim() === '') {
      const topbarResponse = await fetch('../templates/topbar.html');
      if (!topbarResponse.ok) {
        throw new Error(`Failed to fetch topbar.html: ${topbarResponse.status}`);
      }
      const topbarData = await topbarResponse.text();
      topbarElement.innerHTML = topbarData;
    } else {
      console.log('Topbar content already loaded, skipping fetch');
    }

    // Load menus
    const menusResponse = await fetch('../templates/menus.html');
    if (!menusResponse.ok) {
      throw new Error(`Failed to fetch menus.html: ${menusResponse.status}`);
    }
    const menusData = await menusResponse.text();
    menusElement.innerHTML = menusData;

    // Ensure menu links are styled correctly
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    const menuLinks = menusElement.querySelectorAll('.nav-link');
    menuLinks.forEach(link => {
      if (link.getAttribute('href') === currentPage) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });

  } catch (error) {
    console.error('Error in loadLayouts.js:', error);
    // Display error message in the UI
    const topbarElement = document.getElementById('topbar') ||
                         document.getElementById('topbar-placeholder') ||
                         document.getElementById('topbar-container');
    const menusElement = document.getElementById('menus');
    if (topbarElement) {
      topbarElement.innerHTML = '<div class="alert alert-danger">Failed to load topbar</div>';
    }
    if (menusElement) {
      menusElement.innerHTML = '<div class="alert alert-danger">Failed to load menus</div>';
    }
  }
});
