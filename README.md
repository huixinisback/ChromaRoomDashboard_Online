# ChromaRoom Analytics Dashboard

A modern HTML/JavaScript analytics dashboard for ChromaRoom game data.

## Features

- 📊 **6 Comprehensive Analytics Tabs:**
  - 👤 Player Lifetime & Engagement
  - 📍 Area Popularity
  - ⚙️ Feature Usage
  - 👥 Social Connections
  - 🎂 Player Age Groups (with pie chart)
  - 👗 Avatar Item Popularity

- 🎨 **Modern UI:**
  - Beautiful gradient design
  - Responsive layout
  - Interactive tabs
  - Professional metric cards
  - Data tables and visualizations

## How to Use

1. **Open the Dashboard:**
   - Simply open `index.html` in a modern web browser (Chrome, Firefox, Edge, Safari)
   - No server required - works entirely in the browser!

2. **Load Your Data:**
   - Click the file input at the top
   - Select your `datastore.sqlite3` file
   - The dashboard will automatically load and display all analytics

3. **Navigate Tabs:**
   - Click on any tab at the top to view different analytics sections
   - All data is processed and displayed automatically

## Technical Details

- **Technologies Used:**
  - Pure HTML/CSS/JavaScript (no build step required)
  - [sql.js](https://sql.js.org/) for SQLite database reading in the browser
  - [Chart.js](https://www.chartjs.org/) for visualizations
  - Modern CSS with gradients and responsive design

- **Browser Compatibility:**
  - Works in all modern browsers
  - Requires JavaScript enabled
  - No internet connection needed after initial load (CDN resources cached)

## File Structure

```
ChromaRoomDashboard_Online/
├── index.html          # Main dashboard file
├── datastore_*.sqlite3 # Your data files
└── README.md          # This file
```

## Data Format

The dashboard expects SQLite database files with the following structure:
- Table with `datastore_name` column to filter data
- `data_raw` column containing JSON data (for most datastores)
- Or direct column format for some datastores

Supported datastores:
- `PlayerLifetime_V1`
- `AreaTimeRecords_V1`
- `FeatureUsage_V1`
- `FriendConnections_V1`
- `PlayerAgeGroups_V1`
- `AvatarItemUsage_V1`

## Comparison with Python Version

This HTML/JavaScript version provides:
- ✅ Same analytics and visualizations
- ✅ No Python installation required
- ✅ Works directly in browser
- ✅ No server needed
- ✅ Modern, responsive UI
- ✅ All 6 tabs fully functional

## Notes

- The dashboard loads data entirely client-side
- Large database files may take a moment to process
- All calculations are performed in JavaScript
- Data never leaves your browser (privacy-friendly)

## GitHub Pages Deployment

This dashboard can be easily deployed as a static site on GitHub Pages!

### Quick Setup

1. **Create a GitHub repository** and push your code:
   ```bash
   git init
   git add index.html README.md .gitignore
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

2. **Enable GitHub Pages:**
   - Go to your repository on GitHub
   - Click **Settings** → **Pages**
   - Under "Source", select **Deploy from a branch**
   - Choose **main** branch and **/ (root)** folder
   - Click **Save**

3. **Access your site:**
   - Your dashboard will be available at: `https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/`

### Important Notes for GitHub Pages

- **Database files**: SQLite files are excluded via `.gitignore` (they can be large)
- **User upload**: Users will need to upload their own `datastore.sqlite3` file when using the site
- **Privacy**: All processing happens in the browser - no data is sent to any server
- **No backend needed**: This is a pure static site with no server-side requirements

### File Size Considerations

- GitHub has a 100MB file size limit for individual files
- If your database files are large, consider:
  - Keeping them in `.gitignore` (users upload their own)
  - Or using Git LFS for large files
  - Or hosting sample data files separately

