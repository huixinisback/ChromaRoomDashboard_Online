import json
import os
import sqlite3

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="ChromaRoom Dashboard",
    page_icon="📊",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("📊 ChromaRoom Dashboard")
st.sidebar.markdown("---")

# Main title
st.title("📊 ChromaRoom Analytics Dashboard")
st.markdown("---")

# --- Helper Functions ---


def load_from_consolidated_datastore(datastore_name: str, parse_func=None, column_mapper=None):
    """
    Load data from consolidated datastore.sqlite3 file.
    
    Args:
        datastore_name: The datastore name to filter by (e.g., 'PlayerLifetime_V1')
        parse_func: Optional function to parse records from JSON (for data_raw format)
        column_mapper: Optional function to map column names (for direct column format)
    
    Returns:
        DataFrame or None
    """
    # Try to find datastore.sqlite3 file (with timestamp or without)
    datastore_files = []
    for f in os.listdir('.'):
        if f.startswith('datastore') and f.endswith('.sqlite3'):
            datastore_files.append(f)
    
    if not datastore_files:
        return None
    
    # Use the most recent datastore file if multiple exist
    datastore_file = sorted(datastore_files)[-1]
    
    try:
        conn = sqlite3.connect(datastore_file)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        if not tables:
            conn.close()
            return None
        
        # Use the first table found
        table_name = tables[0][0]
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Check if datastore_name column exists
        if 'datastore_name' not in columns:
            conn.close()
            return None
        
        # Filter by datastore_name
        query = f"SELECT * FROM {table_name} WHERE datastore_name = ?"
        df = pd.read_sql_query(query, conn, params=(datastore_name,))
        
        if df.empty:
            conn.close()
            return None
        
        # Check if data_raw column exists
        if 'data_raw' in columns:
            # Parse JSON from data_raw column
            if parse_func:
                records = []
                for idx, row in df.iterrows():
                    try:
                        data_str = row['data_raw']
                        if isinstance(data_str, str):
                            obj = json.loads(data_str)
                            parsed = parse_func(obj)
                            if parsed:
                                records.append(parsed)
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
                
                if records:
                    df = pd.DataFrame.from_records(records)
                else:
                    df = pd.DataFrame()
            else:
                # Default parsing: try to extract common fields from JSON
                records = []
                for idx, row in df.iterrows():
                    try:
                        data_str = row['data_raw']
                        if isinstance(data_str, str):
                            records.append(json.loads(data_str))
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
                
                if records:
                    df = pd.DataFrame.from_records(records)
                else:
                    df = pd.DataFrame()
        else:
            # Direct column format - use column mapper if provided
            if column_mapper:
                df = column_mapper(df)
        
        conn.close()
        
        if df.empty:
            return None
        
        return df
        
    except Exception as e:
        st.warning(f"Error reading consolidated datastore: {e}")
        return None


def parse_avatar_item_json(obj):
    """Parse avatar item record from JSON object."""
    return {
        "assetId": obj.get("assetId"),
        "assetName": obj.get("assetName", "Unknown"),
        "creatorName": obj.get("creatorName", "Unknown"),
        "timesSeen": obj.get("timesSeen", 0),
    }


def map_avatar_item_columns(df):
    """Map column names for avatar item data."""
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'assetid' in col_lower or 'asset_id' in col_lower:
            column_mapping[col] = 'assetId'
        elif 'assetname' in col_lower or 'asset_name' in col_lower:
            column_mapping[col] = 'assetName'
        elif 'creatorname' in col_lower or 'creator_name' in col_lower:
            column_mapping[col] = 'creatorName'
        elif 'timeseen' in col_lower or 'times_seen' in col_lower:
            column_mapping[col] = 'timesSeen'
    
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist, fill missing with defaults
    if 'assetName' not in df.columns:
        df['assetName'] = 'Unknown'
    if 'creatorName' not in df.columns:
        df['creatorName'] = 'Unknown'
    if 'timesSeen' not in df.columns:
        df['timesSeen'] = 0
    
    return df


def load_avatar_item_data():
    """Load avatar item tracking data from consolidated datastore."""
    df = load_from_consolidated_datastore("AvatarItemUsage_V1", parse_avatar_item_json, map_avatar_item_columns)
    
    if df is None or df.empty:
        return None
    
    # Cast columns
    df["timesSeen"] = pd.to_numeric(df["timesSeen"], errors="coerce").fillna(0)
    df["assetId"] = pd.to_numeric(df["assetId"], errors="coerce")
    
    # Drop rows with missing assetId
    df = df.dropna(subset=["assetId"])
    
    if df.empty:
        return None
    
    # Sort by timesSeen descending
    df = df.sort_values("timesSeen", ascending=False)
    
    return df


def parse_player_lifetime_json(obj):
    """Parse player lifetime record from JSON object."""
    return {
        "userId": obj.get("userId"),
        "timesGameVisited": obj.get("timesGameVisited", 0),
        "totalDurationSec": obj.get("totalDurationSec", 0),
    }


def map_player_lifetime_columns(df):
    """Map column names for player lifetime data."""
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'userid' in col_lower or 'user_id' in col_lower:
            column_mapping[col] = 'userId'
        elif 'timesgamevisited' in col_lower or 'times_game_visited' in col_lower:
            column_mapping[col] = 'timesGameVisited'
        elif 'totaldurationsec' in col_lower or 'total_duration_sec' in col_lower:
            column_mapping[col] = 'totalDurationSec'
    
    return df.rename(columns=column_mapping)


def load_player_lifetime_data():
    """Load player lifetime data from consolidated datastore."""
    df = load_from_consolidated_datastore("PlayerLifetime_V1", parse_player_lifetime_json, map_player_lifetime_columns)
    
    if df is None or df.empty:
        return None
    
    # Cast columns to numeric
    if 'timesGameVisited' in df.columns:
        df['timesGameVisited'] = pd.to_numeric(df['timesGameVisited'], errors="coerce").fillna(0)
    if 'totalDurationSec' in df.columns:
        df['totalDurationSec'] = pd.to_numeric(df['totalDurationSec'], errors="coerce").fillna(0)
    
    # Calculate average time per visit (avoid division by zero)
    df['avgTimePerVisit'] = df.apply(
        lambda row: row['totalDurationSec'] / row['timesGameVisited'] if row['timesGameVisited'] > 0 else 0,
        axis=1
    )
    
    return df


def parse_feature_usage_json(obj):
    """Parse feature usage record from JSON object."""
    return {
        "userId": obj.get("userId"),
        "featureId": obj.get("featureId"),
        "timesUsed": obj.get("timesUsed", 0),
        "totalDurationSec": obj.get("totalDurationSec", 0),
    }


def map_feature_usage_columns(df):
    """Map column names for feature usage data."""
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'userid' in col_lower or 'user_id' in col_lower:
            column_mapping[col] = 'userId'
        elif 'featureid' in col_lower or 'feature_id' in col_lower:
            column_mapping[col] = 'featureId'
        elif 'timesused' in col_lower or 'times_used' in col_lower:
            column_mapping[col] = 'timesUsed'
        elif 'totaldurationsec' in col_lower or 'total_duration_sec' in col_lower:
            column_mapping[col] = 'totalDurationSec'
    
    return df.rename(columns=column_mapping)


def load_feature_usage_data():
    """Load feature usage data from consolidated datastore."""
    df = load_from_consolidated_datastore("FeatureUsage_V1", parse_feature_usage_json, map_feature_usage_columns)
    
    if df is None or df.empty:
        return None
    
    # Cast columns to numeric
    if 'timesUsed' in df.columns:
        df['timesUsed'] = pd.to_numeric(df['timesUsed'], errors="coerce").fillna(0)
    if 'totalDurationSec' in df.columns:
        df['totalDurationSec'] = pd.to_numeric(df['totalDurationSec'], errors="coerce").fillna(0)
    
    return df


def parse_player_age_group_json(obj):
    """Parse player age group record from JSON object."""
    return {
        "userId": obj.get("userId"),
        "ageGroup": obj.get("ageGroup", "Unknown"),
    }


def map_player_age_group_columns(df):
    """Map column names for player age group data."""
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'userid' in col_lower or 'user_id' in col_lower:
            column_mapping[col] = 'userId'
        elif 'agegroup' in col_lower or 'age_group' in col_lower:
            column_mapping[col] = 'ageGroup'
    
    return df.rename(columns=column_mapping)


def load_player_age_group_data():
    """Load player age group data from consolidated datastore."""
    df = load_from_consolidated_datastore("PlayerAgeGroups_V1", parse_player_age_group_json, map_player_age_group_columns)
    
    if df is None or df.empty:
        return None
    
    # Ensure ageGroup column exists
    if 'ageGroup' not in df.columns:
        df['ageGroup'] = 'Unknown'
    
    # Normalize ageGroup values
    df['ageGroup'] = df['ageGroup'].astype(str)
    df['ageGroup'] = df['ageGroup'].replace({
        'under13': 'Under13',
        '13plus': '13Plus',
        '13+': '13Plus',
        '<13': 'Under13',
    })
    
    # Only keep valid age groups
    valid_groups = ['Under13', '13Plus', 'Unknown']
    df = df[df['ageGroup'].isin(valid_groups)]
    
    return df


def parse_friend_connection_json(obj):
    """Parse friend connection record from JSON object."""
    return {
        "userId": obj.get("userId"),
        "friendsInLastGame": obj.get("friendsInLastGame", 0),
        "maxFriendsInOneGame": obj.get("maxFriendsInOneGame", 0),
        "totalTimeWithFriends": obj.get("totalTimeWithFriends", 0),
        "longestContinuousFriendSession": obj.get("longestContinuousFriendSession", 0),
    }


def map_friend_connection_columns(df):
    """Map column names for friend connection data."""
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'userid' in col_lower or 'user_id' in col_lower:
            column_mapping[col] = 'userId'
        elif 'friendsinlastgame' in col_lower or 'friends_in_last_game' in col_lower:
            column_mapping[col] = 'friendsInLastGame'
        elif 'maxfriendsinonegame' in col_lower or 'max_friends_in_one_game' in col_lower:
            column_mapping[col] = 'maxFriendsInOneGame'
        elif 'totaltimewithfriends' in col_lower or 'total_time_with_friends' in col_lower:
            column_mapping[col] = 'totalTimeWithFriends'
        elif 'longestcontinuousfriendsession' in col_lower or 'longest_continuous_friend_session' in col_lower:
            column_mapping[col] = 'longestContinuousFriendSession'
    
    return df.rename(columns=column_mapping)


def load_friend_connection_data():
    """Load friend connection data from consolidated datastore."""
    df = load_from_consolidated_datastore("FriendConnections_V1", parse_friend_connection_json, map_friend_connection_columns)
    
    if df is None or df.empty:
        return None
    
    # Cast columns to numeric
    numeric_cols = ["friendsInLastGame", "maxFriendsInOneGame", "totalTimeWithFriends", "longestContinuousFriendSession"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    return df


def parse_area_tracking_json(obj):
    """Parse area tracking record from JSON object."""
    return {
        "areaId": obj.get("areaId"),
        "totalDurationSec": obj.get("totalDurationSec"),
        "timesVisited": obj.get("timesVisited"),
        "lastExitedIso": obj.get("lastExitedIso"),
        "firstEnteredIso": obj.get("firstEnteredIso"),
    }


def map_area_tracking_columns(df):
    """Map column names for area tracking data."""
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'areaid' in col_lower or 'area_id' in col_lower:
            column_mapping[col] = 'areaId'
        elif 'totaldurationsec' in col_lower or 'total_duration_sec' in col_lower:
            column_mapping[col] = 'totalDurationSec'
        elif 'timesvisited' in col_lower or 'times_visited' in col_lower:
            column_mapping[col] = 'timesVisited'
        elif 'lastexitediso' in col_lower or 'last_exited_iso' in col_lower:
            column_mapping[col] = 'lastExitedIso'
        elif 'firstenterediso' in col_lower or 'first_entered_iso' in col_lower:
            column_mapping[col] = 'firstEnteredIso'
    
    return df.rename(columns=column_mapping)


def load_area_data():
    """Load area tracking data from consolidated datastore."""
    df = load_from_consolidated_datastore("AreaTimeRecords_V1", parse_area_tracking_json, map_area_tracking_columns)
    
    if df is None or df.empty:
        return None
    
    # Cast columns
    num_cols = ["totalDurationSec", "timesVisited"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    if "lastExitedIso" in df.columns:
        df["lastExitedIso"] = pd.to_datetime(df["lastExitedIso"], errors="coerce", utc=True)
    if "firstEnteredIso" in df.columns:
        df["firstEnteredIso"] = pd.to_datetime(df["firstEnteredIso"], errors="coerce", utc=True)
    
    # Drop rows lacking areaId
    valid = df.dropna(subset=["areaId"])
    return valid if not valid.empty else None


# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "👤 Player Lifetime & Engagement",
    "📍 Area Popularity",
    "⚙️ Feature Usage",
    "👥 Social Connections",
    "🎂 Player Age Groups",
    "👗 Avatar Item Popularity"
])

# Tab 1: Player Lifetime and Engagement
with tab1:
    st.header("👤 Player Lifetime and Engagement")
    st.subheader("Datastore: `PlayerLifetime_V1`")
    
    st.markdown("### Purpose")
    st.markdown("""
    This metric tracks player retention and total engagement duration.
    It provides insight into:
    - How frequently players return to the game
    - The average time spent per session and across all sessions
    - Long-term engagement trends that can inform content updates and reward pacing
    """)
    
    st.markdown("### Schema")
    st.code("""
{
  userId = number,               -- Roblox user ID
  firstPlayedIso = string?,      -- ISO UTC time of first join
  lastPlayedIso = string?,       -- ISO UTC time of most recent exit
  timesGameVisited = number,     -- number of sessions joined
  totalDurationSec = number,     -- cumulative seconds spent across all sessions
  v = 1
}
    """, language="json")
    
    st.markdown("---")
    st.markdown("### Analytics & Visualizations")
    
    # Load player lifetime data
    df = load_player_lifetime_data()
    
    if df is None:
        st.warning("⚠️ No player lifetime data found. Please ensure the consolidated datastore.sqlite3 file exists with 'PlayerLifetime_V1' datastore.")
    else:
        st.markdown(f"**Total Players:** {len(df):,}")
        st.markdown("---")
        
        # timesGameVisited analytics
        if "timesGameVisited" in df.columns:
            st.markdown("### 🎮 Times Game Visited")
            times_visited = df[df["timesGameVisited"] > 0]["timesGameVisited"].dropna()
            
            if len(times_visited) > 0:
                # Highest frequency (mode)
                mode_result = times_visited.mode()
                highest_freq = mode_result.iloc[0] if len(mode_result) > 0 else None
                freq_count = (times_visited == highest_freq).sum() if highest_freq is not None else 0
                
                # Top 5 data points
                top_5_visited = times_visited.nlargest(5).tolist()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("**Top 5 Data Points:**")
                    for i, val in enumerate(top_5_visited, 1):
                        st.markdown(f"{i}. {int(val):,}")
                with col2:
                    st.metric("Highest Frequency", f"{int(highest_freq)}" if highest_freq is not None else "N/A",
                             delta=f"{freq_count} occurrences" if highest_freq is not None else None)
                with col3:
                    st.metric("Mean", f"{times_visited.mean():.2f}")
                with col4:
                    st.metric("Median", f"{int(times_visited.median())}")
            else:
                st.info("No data available for times game visited.")
        
        st.markdown("---")
        
        # totalDurationSec analytics
        if "totalDurationSec" in df.columns:
            st.markdown("### ⏱️ Total Duration (Seconds)")
            total_duration = df[df["totalDurationSec"] > 0]["totalDurationSec"].dropna()
            
            if len(total_duration) > 0:
                # Highest frequency (mode)
                mode_result = total_duration.mode()
                highest_freq = mode_result.iloc[0] if len(mode_result) > 0 else None
                freq_count = (total_duration == highest_freq).sum() if highest_freq is not None else 0
                
                # Top 5 data points
                top_5_duration = total_duration.nlargest(5).tolist()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("**Top 5 Data Points:**")
                    for i, val in enumerate(top_5_duration, 1):
                        # Convert to readable format
                        hours = int(val // 3600)
                        minutes = int((val % 3600) // 60)
                        seconds = int(val % 60)
                        if hours > 0:
                            duration_str = f"{hours}h {minutes}m {seconds}s"
                        elif minutes > 0:
                            duration_str = f"{minutes}m {seconds}s"
                        else:
                            duration_str = f"{seconds}s"
                        st.markdown(f"{i}. {duration_str} ({int(val):,}s)")
                with col2:
                    # Format highest frequency
                    if highest_freq is not None:
                        hf_hours = int(highest_freq // 3600)
                        hf_minutes = int((highest_freq % 3600) // 60)
                        hf_seconds = int(highest_freq % 60)
                        if hf_hours > 0:
                            hf_str = f"{hf_hours}h {hf_minutes}m {hf_seconds}s"
                        elif hf_minutes > 0:
                            hf_str = f"{hf_minutes}m {hf_seconds}s"
                        else:
                            hf_str = f"{hf_seconds}s"
                        st.metric("Highest Frequency", hf_str,
                                 delta=f"{freq_count} occurrences")
                    else:
                        st.metric("Highest Frequency", "N/A")
                with col3:
                    avg_seconds = total_duration.mean()
                    avg_hours = int(avg_seconds // 3600)
                    avg_minutes = int((avg_seconds % 3600) // 60)
                    avg_secs = int(avg_seconds % 60)
                    if avg_hours > 0:
                        avg_str = f"{avg_hours}h {avg_minutes}m {avg_secs}s"
                    elif avg_minutes > 0:
                        avg_str = f"{avg_minutes}m {avg_secs}s"
                    else:
                        avg_str = f"{avg_secs}s"
                    st.metric("Mean", avg_str, delta=f"{avg_seconds:.2f} seconds")
                with col4:
                    median_seconds = total_duration.median()
                    med_hours = int(median_seconds // 3600)
                    med_minutes = int((median_seconds % 3600) // 60)
                    med_secs = int(median_seconds % 60)
                    if med_hours > 0:
                        med_str = f"{med_hours}h {med_minutes}m {med_secs}s"
                    elif med_minutes > 0:
                        med_str = f"{med_minutes}m {med_secs}s"
                    else:
                        med_str = f"{med_secs}s"
                    st.metric("Median", med_str, delta=f"{int(median_seconds):,} seconds")
            else:
                st.info("No data available for total duration.")
        
        st.markdown("---")
        
        # Average time per visit analytics
        if "avgTimePerVisit" in df.columns:
            st.markdown("### 📊 Average Time Per Visit")
            avg_time = df[df["avgTimePerVisit"] > 0]["avgTimePerVisit"].dropna()
            
            if len(avg_time) > 0:
                # Highest 5 frequency (top 5 most common values)
                value_counts = avg_time.value_counts().head(5)
                
                # Median 5 (middle 5 values when sorted)
                sorted_avg = avg_time.sort_values()
                median_5_start = len(sorted_avg) // 2 - 2
                median_5_end = median_5_start + 5
                median_5 = sorted_avg.iloc[max(0, median_5_start):median_5_end].tolist()
                
                # Lowest 5
                lowest_5 = avg_time.nsmallest(5).tolist()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Highest 5 Frequency:**")
                    if len(value_counts) > 0:
                        for val, count in value_counts.items():
                            hours = int(val // 3600)
                            minutes = int((val % 3600) // 60)
                            seconds = int(val % 60)
                            if hours > 0:
                                val_str = f"{hours}h {minutes}m {seconds}s"
                            elif minutes > 0:
                                val_str = f"{minutes}m {seconds}s"
                            else:
                                val_str = f"{seconds}s"
                            st.markdown(f"- {val_str}: {int(count)} occurrences")
                    else:
                        st.markdown("N/A")
                
                with col2:
                    st.markdown("**Median 5:**")
                    if len(median_5) > 0:
                        # Get occurrence counts for median 5 values
                        all_value_counts = avg_time.value_counts()
                        for i, val in enumerate(median_5, 1):
                            hours = int(val // 3600)
                            minutes = int((val % 3600) // 60)
                            seconds = int(val % 60)
                            if hours > 0:
                                val_str = f"{hours}h {minutes}m {seconds}s"
                            elif minutes > 0:
                                val_str = f"{minutes}m {seconds}s"
                            else:
                                val_str = f"{seconds}s"
                            count = int(all_value_counts.get(val, 0))
                            st.markdown(f"{i}. {val_str}: {count} occurrences")
                    else:
                        st.markdown("N/A")
                
                with col3:
                    st.markdown("**Lowest 5:**")
                    if len(lowest_5) > 0:
                        # Get occurrence counts for lowest 5 values
                        all_value_counts = avg_time.value_counts()
                        for i, val in enumerate(lowest_5, 1):
                            hours = int(val // 3600)
                            minutes = int((val % 3600) // 60)
                            seconds = int(val % 60)
                            if hours > 0:
                                val_str = f"{hours}h {minutes}m {seconds}s"
                            elif minutes > 0:
                                val_str = f"{minutes}m {seconds}s"
                            else:
                                val_str = f"{seconds}s"
                            count = int(all_value_counts.get(val, 0))
                            st.markdown(f"{i}. {val_str}: {count} occurrences")
                    else:
                        st.markdown("N/A")
            else:
                st.info("No data available for average time per visit.")

# Tab 2: Area Popularity and Player Preferences
with tab2:
    st.header("📍 Area Popularity and Player Preferences")
    st.subheader("Datastore: `AreaTimeRecords_V1`")
    
    st.markdown("### Purpose")
    st.markdown("""
    Tracks time spent in different areas or themed rooms within the game.
    
    **Why it matters:**
    This helps identify which environments or themes are most appealing. Areas with high dwell time can guide future aesthetic or gameplay direction, while low-engagement zones may need rework or additional incentives.
    """)
    
    st.markdown("### Schema")
    st.code("""
{
  uniqueId = "userid_areaid",
  userId = number,
  areaId = string,
  firstEnteredIso = string?,
  lastExitedIso = string?,
  totalDurationSec = number,
  timesVisited = number,
  v = 1
}
    """, language="json")
    
    st.markdown("---")
    st.markdown("### Analytics & Visualizations")
    
    # Load area data
    df = load_area_data()
    
    if df is None:
        st.warning("⚠️ No area data found. Please ensure the consolidated datastore.sqlite3 file exists with 'AreaTimeRecords_V1' datastore.")
    else:
        # Get unique area IDs
        area_ids = sorted(df['areaId'].unique())
        
        st.markdown(f"**Total Areas:** {len(area_ids)} | **Total Records:** {len(df)}")
        st.markdown("---")
        
        # Display each area in its own card
        for area_id in area_ids:
            area_data = df[df['areaId'] == area_id]
            times_visited = area_data['timesVisited'].dropna()
            
            if len(times_visited) == 0:
                continue
            
            # Calculate metrics
            total_times_visited = times_visited.sum()
            mean_times_visited = times_visited.mean()
            median_times_visited = times_visited.median()
            max_times_visited = times_visited.max()
            
            # Most common (mode)
            mode_result = times_visited.mode()
            most_common = mode_result.iloc[0] if len(mode_result) > 0 else None
            most_common_count = (times_visited == most_common).sum() if most_common is not None else 0
            
            # Count frequency of each visit count
            visit_counts = times_visited.value_counts().sort_values(ascending=False)
            
            # Top 5 most common visit counts (highest frequency)
            top_5_visit_counts = visit_counts.head(5)
            top_5_list = []
            if len(top_5_visit_counts) > 0:
                top_5_list = [f"{int(count)} visits: {int(freq)} occurrences" for count, freq in zip(top_5_visit_counts.index, top_5_visit_counts.values)]
            
            # Bottom 5 least common visit counts (lowest frequency)
            bottom_5_visit_counts = visit_counts.tail(5).sort_values(ascending=True)
            bottom_5_list = []
            if len(bottom_5_visit_counts) > 0:
                bottom_5_list = [f"{int(count)} visits: {int(freq)} occurrences" for count, freq in zip(bottom_5_visit_counts.index, bottom_5_visit_counts.values)]
            
            # Last entered time (most recent firstEnteredIso)
            last_entered = area_data['firstEnteredIso'].max()
            if pd.notna(last_entered):
                last_entered_str = pd.to_datetime(last_entered, utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")
            else:
                last_entered_str = "N/A"
            
            # Create card for this area
            with st.container():
                st.markdown(f"### 🏷️ {area_id}")
                
                # Create metric columns
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Times Visited", f"{int(total_times_visited):,}")
                
                with col2:
                    st.metric("Mean Times Visited", f"{mean_times_visited:.2f}")
                
                with col3:
                    st.metric("Median Times Visited", f"{int(median_times_visited)}")
                
                with col4:
                    st.metric("Max Times Visited", f"{int(max_times_visited)}")
                
                with col5:
                    if most_common is not None:
                        st.metric("Most Common", f"{int(most_common)}", 
                                 delta=f"{int(most_common_count)} occurrences")
                    else:
                        st.metric("Most Common", "N/A")
                
                # Second row - Top 5 and Bottom 5 visit counts
                col6, col7 = st.columns(2)
                
                with col6:
                    st.markdown("**Top 5 Most Common Visit Counts:**")
                    if len(top_5_list) > 0:
                        for item in top_5_list:
                            st.markdown(f"- {item}")
                    else:
                        st.markdown("N/A")
                
                with col7:
                    st.markdown("**Bottom 5 Least Common Visit Counts:**")
                    if len(bottom_5_list) > 0:
                        for item in bottom_5_list:
                            st.markdown(f"- {item}")
                    else:
                        st.markdown("N/A")
                
                # Third row - Last entered time
                st.metric("Last Entered Time", last_entered_str)
                
                st.markdown("---")

# Tab 3: Feature Usage and Interaction Patterns
with tab3:
    st.header("⚙️ Feature Usage and Interaction Patterns")
    st.subheader("Datastore: `FeatureUsage_V1`")
    
    st.markdown("### Purpose")
    st.markdown("""
    Tracks engagement with specific game features (e.g., Try-On GUI, inventory menu, mini-games).
    
    **Why it matters:**
    Reveals which features are most popular or underutilized. This data can prioritize development focus and UX improvements, particularly useful for evaluating new feature launches or updates.
    """)
    
    st.markdown("### Schema")
    st.code("""
{
  uniqueKey = "userid_featureId",
  userId = number,
  featureId = string,
  firstUsedIso = string?,
  lastUsedIso = string?,
  totalDurationSec = number,
  timesUsed = number,
  v = 1
}
    """, language="json")
    
    st.markdown("---")
    st.markdown("### Analytics & Visualizations")
    
    # Load feature usage data
    df = load_feature_usage_data()
    
    if df is None:
        st.warning("⚠️ No feature usage data found. Please ensure the consolidated datastore.sqlite3 file exists with 'FeatureUsage_V1' datastore.")
    else:
        # Get unique features
        if 'featureId' not in df.columns:
            st.error("Feature ID column not found in data.")
        else:
            features = sorted(df['featureId'].dropna().unique())
            
            if len(features) == 0:
                st.info("No features found in the data.")
            else:
                st.markdown(f"**Total Features:** {len(features)} | **Total Records:** {len(df)}")
                st.markdown("---")
                
                # Create a card for each feature
                for feature_id in features:
                    feature_data = df[df['featureId'] == feature_id]
                    
                    # Calculate metrics
                    total_times_used = feature_data['timesUsed'].sum() if 'timesUsed' in feature_data.columns else 0
                    total_duration_sec = feature_data['totalDurationSec'].sum() if 'totalDurationSec' in feature_data.columns else 0
                    zero_usage_count = len(feature_data[feature_data['timesUsed'] == 0]) if 'timesUsed' in feature_data.columns else 0
                    total_players = len(feature_data)
                    
                    # Get duration data for frequency analysis
                    duration_data = feature_data[feature_data['totalDurationSec'] > 0]['totalDurationSec'].dropna()
                    
                    # Create card
                    with st.container():
                        st.markdown(f"### 🎮 {feature_id}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Times Used", f"{int(total_times_used):,}")
                        
                        with col2:
                            # Convert seconds to minutes (rounded)
                            total_duration_minutes = round(total_duration_sec / 60)
                            st.metric("Total Duration", f"{total_duration_minutes:,} minutes", 
                                     delta=f"{int(total_duration_sec):,} seconds")
                        
                        with col3:
                            st.metric("Players with 0 Uses", f"{zero_usage_count:,}", 
                                     delta=f"{zero_usage_count/total_players*100:.1f}%" if total_players > 0 else "0%")
                        
                        with col4:
                            players_with_usage = total_players - zero_usage_count
                            st.metric("Players Who Used", f"{players_with_usage:,}", 
                                     delta=f"{players_with_usage/total_players*100:.1f}%" if total_players > 0 else "0%")
                        
                        # Total Duration Frequency Analysis
                        if len(duration_data) > 0:
                            # Round to nearest 15 seconds
                            # Round seconds to nearest 15: round(seconds / 15) * 15
                            duration_rounded_15s = (duration_data / 15).round() * 15
                            
                            # Filter out zeros
                            duration_rounded_15s = duration_rounded_15s[duration_rounded_15s > 0]
                            
                            if len(duration_rounded_15s) > 0:
                                # Group by unique 15-second intervals and count occurrences
                                seconds_counts = duration_rounded_15s.value_counts().sort_index()
                                
                                # Convert to list of (seconds, count) tuples for easier manipulation
                                intervals_with_counts = [(int(secs), int(count)) for secs, count in seconds_counts.items()]
                                
                                if len(intervals_with_counts) > 0:
                                    # Top 5 frequency (most common 15s intervals)
                                    top_5 = sorted(intervals_with_counts, key=lambda x: x[1], reverse=True)[:5]
                                    
                                    # Median 5 (middle 5 unique intervals when sorted by occurrence count)
                                    sorted_by_occurrence = sorted(intervals_with_counts, key=lambda x: x[1])
                                    median_5_start = len(sorted_by_occurrence) // 2 - 2
                                    median_5_end = median_5_start + 5
                                    median_5 = sorted_by_occurrence[max(0, median_5_start):median_5_end]
                                    
                                    # Lowest 5 frequency (least common 15s intervals, non-zero only)
                                    # Filter to only non-zero counts, then sort by count ascending
                                    non_zero_intervals = [(secs, count) for secs, count in intervals_with_counts if count > 0]
                                    lowest_5 = sorted(non_zero_intervals, key=lambda x: x[1])[:5]
                                    
                                    col5, col6, col7 = st.columns(3)
                                    
                                    with col5:
                                        st.markdown("**Total Duration - Top 5 Frequency:**")
                                        if len(top_5) > 0:
                                            for secs, count in top_5:
                                                mins = secs // 60
                                                remaining_secs = secs % 60
                                                if mins > 0:
                                                    st.markdown(f"- {secs}s ({mins}min {remaining_secs}s): {count} occurrences")
                                                else:
                                                    st.markdown(f"- {secs}s: {count} occurrences")
                                        else:
                                            st.markdown("N/A")
                                    
                                    with col6:
                                        st.markdown("**Total Duration - Median 5:**")
                                        if len(median_5) > 0:
                                            for i, (secs, count) in enumerate(median_5, 1):
                                                mins = secs // 60
                                                remaining_secs = secs % 60
                                                if mins > 0:
                                                    st.markdown(f"{i}. {secs}s ({mins}min {remaining_secs}s): {count} occurrences")
                                                else:
                                                    st.markdown(f"{i}. {secs}s: {count} occurrences")
                                        else:
                                            st.markdown("N/A")
                                    
                                    with col7:
                                        st.markdown("**Total Duration - Lowest 5:**")
                                        if len(lowest_5) > 0:
                                            for i, (secs, count) in enumerate(lowest_5, 1):
                                                mins = secs // 60
                                                remaining_secs = secs % 60
                                                if mins > 0:
                                                    st.markdown(f"{i}. {secs}s ({mins}min {remaining_secs}s): {count} occurrences")
                                                else:
                                                    st.markdown(f"{i}. {secs}s: {count} occurrences")
                                        else:
                                            st.markdown("N/A")
                                else:
                                    st.info("No duration data available for frequency analysis.")
                            else:
                                st.info("No duration data available for frequency analysis.")
                        
                        st.markdown("---")

# Tab 4: Social Connections and Friend-Based Play
with tab4:
    st.header("👥 Social Connections and Friend-Based Play")
    st.subheader("Datastore: `FriendConnections_V1`")
    
    st.markdown("### Purpose")
    st.markdown("""
    Tracks how many friend connections are in the same server/session and correlates with total duration played.
    Measures the impact of social play on retention and engagement.
    
    **Why it matters:**
    Roblox experiences thrive on social connections. Players who join with friends tend to stay longer and return more often. This metric supports design decisions that encourage multiplayer cooperation and social discovery.
    """)
    
    st.markdown("### Schema")
    st.code("""
{
   userId = number,                         -- Roblox user ID
   friendsInLastGame = number,              -- number of friends in the same server at the end of their session
   maxFriendsInOneGame = number,            -- maximum number of friends seen at once in any session
   totalTimeWithFriends = number,           -- cumulative seconds spent while at least one friend was present
   longestContinuousFriendSession = number, -- longest continuous period with ≥1 friend present (seconds)
   v = 1,                                   -- schema version (fixed at 1)
}
    """, language="json")
    
    st.markdown("---")
    st.markdown("### Analytics & Visualizations")
    
    # Load friend connection data
    df = load_friend_connection_data()
    
    if df is None:
        st.warning("⚠️ No friend connection data found. Please ensure the consolidated datastore.sqlite3 file exists with 'FriendConnections_V1' datastore.")
    else:
        st.markdown(f"**Total Records:** {len(df)}")
        
        # Calculate how many players play with friends
        # A player plays with friends if any of these metrics > 0
        players_with_friends = 0
        if all(col in df.columns for col in ["friendsInLastGame", "maxFriendsInOneGame", "totalTimeWithFriends", "longestContinuousFriendSession"]):
            players_with_friends = len(df[
                (df["friendsInLastGame"] > 0) | 
                (df["maxFriendsInOneGame"] > 0) | 
                (df["totalTimeWithFriends"] > 0) | 
                (df["longestContinuousFriendSession"] > 0)
            ])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Players", f"{len(df):,}")
        with col2:
            st.metric("Players Who Play With Friends", f"{players_with_friends:,}", 
                     delta=f"{players_with_friends/len(df)*100:.1f}%" if len(df) > 0 else "0%")
        
        st.markdown("---")
        
        # longestContinuousFriendSession
        if "longestContinuousFriendSession" in df.columns:
            st.markdown("### 📊 Longest Continuous Friend Session")
            longest_session = df[df["longestContinuousFriendSession"] > 0]["longestContinuousFriendSession"].dropna()
            
            if len(longest_session) > 0:
                # Get value counts for occurrences
                value_counts = longest_session.value_counts()
                
                # Top 5 by occurrence count (sorted by frequency, highest first)
                top_5_by_occurrence = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                
                # Median 5 points (5 points around the median)
                sorted_values = sorted(longest_session.unique())
                median_idx = len(sorted_values) // 2
                median_5_start = max(0, median_idx - 2)
                median_5_end = min(len(sorted_values), median_idx + 3)
                median_5_values = sorted_values[median_5_start:median_5_end]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Top 5 Points:**")
                    for i, (val, count) in enumerate(top_5_by_occurrence, 1):
                        st.markdown(f"{i}. {int(val):,} seconds: {int(count)} occurrences")
                with col2:
                    st.markdown("**Median 5 Points:**")
                    for i, val in enumerate(median_5_values, 1):
                        count = int(value_counts.get(val, 0))
                        st.markdown(f"{i}. {int(val):,} seconds: {count} occurrences")
                with col3:
                    st.metric("Average", f"{longest_session.mean():.2f} seconds")
            else:
                st.info("No data available for longest continuous friend session.")
        
        st.markdown("---")
        
        # maxFriendsInOneGame
        if "maxFriendsInOneGame" in df.columns:
            st.markdown("### 👥 Max Friends In One Game")
            max_friends = df[df["maxFriendsInOneGame"] > 0]["maxFriendsInOneGame"].dropna()
            
            if len(max_friends) > 0:
                # Get value counts for occurrences
                value_counts = max_friends.value_counts()
                
                # Top 5 by occurrence count (sorted by frequency, highest first)
                top_5_by_occurrence = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                
                # Highest frequency (mode)
                mode_result = max_friends.mode()
                highest_freq = mode_result.iloc[0] if len(mode_result) > 0 else None
                freq_count = (max_friends == highest_freq).sum() if highest_freq is not None else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("**Top 5 Points:**")
                    for i, (val, count) in enumerate(top_5_by_occurrence, 1):
                        st.markdown(f"{i}. {int(val)} friends: {int(count)} occurrences")
                with col2:
                    st.metric("Highest Frequency", f"{int(highest_freq)}" if highest_freq is not None else "N/A", 
                             delta=f"({freq_count} occurrences)" if highest_freq is not None else None)
                with col3:
                    st.metric("Median", f"{int(max_friends.median())}")
                with col4:
                    st.metric("Average", f"{max_friends.mean():.2f}")
            else:
                st.info("No data available for max friends in one game.")
        
        st.markdown("---")
        
        # totalTimeWithFriends
        if "totalTimeWithFriends" in df.columns:
            st.markdown("### ⏱️ Total Time With Friends")
            total_time = df[df["totalTimeWithFriends"] > 0]["totalTimeWithFriends"].dropna()
            
            if len(total_time) > 0:
                # Get value counts for occurrences
                value_counts = total_time.value_counts()
                
                # Top 5 by occurrence count (sorted by frequency, highest first)
                top_5_by_occurrence = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Top 5 Points:**")
                    for i, (val, count) in enumerate(top_5_by_occurrence, 1):
                        st.markdown(f"{i}. {int(val):,} seconds: {int(count)} occurrences")
                with col2:
                    st.metric("Median", f"{int(total_time.median()):,} seconds")
                with col3:
                    st.metric("Average", f"{total_time.mean():.2f} seconds")
            else:
                st.info("No data available for total time with friends.")
        
        st.markdown("---")
        
        # friendsInLastGame
        if "friendsInLastGame" in df.columns:
            st.markdown("### 🎮 Friends In Last Game")
            st.caption("More sensitive to recent changes")
            friends_last = df[df["friendsInLastGame"] > 0]["friendsInLastGame"].dropna()
            
            if len(friends_last) > 0:
                # Get value counts for occurrences
                value_counts = friends_last.value_counts()
                
                # Top 5 by occurrence count (sorted by frequency, highest first)
                top_5_by_occurrence = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                
                # Highest frequency (mode)
                mode_result = friends_last.mode()
                highest_freq = mode_result.iloc[0] if len(mode_result) > 0 else None
                freq_count = (friends_last == highest_freq).sum() if highest_freq is not None else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("**Top 5 Points:**")
                    for i, (val, count) in enumerate(top_5_by_occurrence, 1):
                        st.markdown(f"{i}. {int(val)} friends: {int(count)} occurrences")
                with col2:
                    st.metric("Highest Frequency", f"{int(highest_freq)}" if highest_freq is not None else "N/A", 
                             delta=f"({freq_count} occurrences)" if highest_freq is not None else None)
                with col3:
                    st.metric("Median", f"{int(friends_last.median())} friends")
                with col4:
                    st.metric("Average", f"{friends_last.mean():.2f} friends")
            else:
                st.info("No data available for friends in last game.")

# Tab 5: Player Age Group Segmentation
with tab5:
    st.header("🎂 Player Age Group Segmentation")
    st.subheader("Datastore: `PlayerAgeGroups_V1`")
    
    st.markdown("### Purpose")
    st.markdown("""
    To identify the primary age group of our player base and assess whether features such as voice chat and age-appropriate content are suitable for implementation.
    
    **Why it matters:**
    Identifying the dominant player age group enables data-informed decisions on gameplay features and community tools. Age distribution insights help determine whether social features like voice chat will enhance engagement or pose moderation and compliance challenges.
    """)
    
    st.markdown("### Schema")
    st.code("""
{
  userId = number,
  ageGroup = string,      -- "Under13", "13Plus", or "Unknown"
  v = 1
}
    """, language="json")
    
    st.markdown("---")
    st.markdown("### Analytics & Visualizations")
    
    # Load player age group data
    df = load_player_age_group_data()
    
    if df is None:
        st.warning("⚠️ No player age group data found. Please ensure the consolidated datastore.sqlite3 file exists with 'PlayerAgeGroups_V1' datastore.")
    else:
        # Count age groups
        age_group_counts = df['ageGroup'].value_counts()
        total_players = len(df)
        
        # Calculate percentages
        age_group_percentages = (age_group_counts / total_players * 100).round(2)
        
        # Ensure all three groups are represented
        all_groups = ['Under13', '13Plus', 'Unknown']
        for group in all_groups:
            if group not in age_group_counts.index:
                age_group_counts[group] = 0
                age_group_percentages[group] = 0.0
        
        # Sort by standard order
        age_group_counts = age_group_counts.reindex(all_groups, fill_value=0)
        age_group_percentages = age_group_percentages.reindex(all_groups, fill_value=0.0)
        
        st.markdown(f"**Total Players:** {total_players:,}")
        st.markdown("---")
        
        # Summary metrics at the top
        st.markdown("### 📊 Age Group Distribution")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Under 13", f"{int(age_group_counts['Under13']):,}", 
                     delta=f"{age_group_percentages['Under13']:.2f}%")
        with col2:
            st.metric("13 Plus", f"{int(age_group_counts['13Plus']):,}", 
                     delta=f"{age_group_percentages['13Plus']:.2f}%")
        with col3:
            st.metric("Unknown", f"{int(age_group_counts['Unknown']):,}", 
                     delta=f"{age_group_percentages['Unknown']:.2f}%")
        
        st.markdown("---")
        
        # Pie chart with legend below
        non_zero_groups = age_group_counts[age_group_counts > 0]
        
        if len(non_zero_groups) > 0:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # Define colors for each group
                colors = {
                    'Under13': '#FF6B6B',
                    '13Plus': '#4ECDC4',
                    'Unknown': '#95A5A6'
                }
                chart_colors = [colors.get(group, '#95A5A6') for group in non_zero_groups.index]
                
                # Create pie chart
                wedges, texts, autotexts = ax.pie(
                    non_zero_groups.values,
                    labels=non_zero_groups.index,
                    autopct='%1.1f%%',
                    colors=chart_colors,
                    startangle=90,
                    textprops={'fontsize': 12, 'fontweight': 'bold'},
                    pctdistance=0.85
                )
                
                # Enhance text visibility
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(11)
                
                # Make labels more readable
                for text in texts:
                    text.set_fontsize(12)
                    text.set_fontweight('bold')
                
                ax.set_title('Player Age Group Distribution', fontsize=14, fontweight='bold', pad=20)
                
                # Equal aspect ratio ensures that pie is drawn as a circle
                ax.axis('equal')
                
                # Add legend
                legend_labels = [f"{group}: {int(age_group_counts[group]):,} ({age_group_percentages[group]:.2f}%)" 
                                for group in non_zero_groups.index]
                ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
                
                st.pyplot(fig, clear_figure=True)
        else:
            st.info("No age group data to display in chart.")

# Tab 6: Avatar Item Popularity and Style Insights
with tab6:
    st.header("👗 Avatar Item Popularity and Style Insights")
    st.subheader("Datastore: `AvatarItemUsage_V1`")
    
    st.markdown("### Purpose")
    st.markdown("""
    Tracks the most commonly worn avatar items seen in-game.
    
    **Why it matters:**
    Provides cultural insight into player style preferences, enabling in-game item recommendations, themed events, and promotional collaborations that resonate with the player base.
    """)
    
    st.markdown("### Schema")
    st.code("""
{
   assetId = number,           -- ID of the worn catalog item
   assetName = string?,        -- optional (fetched from MarketplaceService)
   creatorName = string?,      -- optional
   timesSeen = number,         -- number of players wearing this item
   v = 1,                      -- schema version
}
    """, language="json")
    
    st.markdown("---")
    st.markdown("### Analytics & Visualizations")
    
    # Load avatar item data
    df = load_avatar_item_data()
    
    if df is None:
        st.warning("⚠️ No avatar item data found. Please ensure the consolidated datastore.sqlite3 file exists with 'AvatarItemUsage_V1' datastore.")
    else:
        # Get top 100 (all items)
        top_100 = df.head(100)
        
        st.markdown(f"**Total Assets:** {len(df)} | **Showing Top 100**")
        st.markdown("---")
        
        # Create display table with requested columns
        display_df = top_100[["assetName", "creatorName", "timesSeen", "assetId"]].copy()
        display_df.columns = ["Asset Name", "Creator", "Times Seen", "Asset ID"]
        
        # Format the display
        display_df["Times Seen"] = display_df["Times Seen"].astype(int)
        display_df["Asset ID"] = display_df["Asset ID"].astype(int)
        
        # Display table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Assets", f"{len(df):,}")
        with col2:
            st.metric("Total Times Seen", f"{int(df['timesSeen'].sum()):,}")
        with col3:
            st.metric("Average Times Seen", f"{df['timesSeen'].mean():.2f}")
        with col4:
            st.metric("Most Popular Item", f"{int(df.iloc[0]['timesSeen'])}" if len(df) > 0 else "N/A")
        
        st.markdown("---")
        
        # Top 100 NOT from Roblox
        st.markdown("### Top 100 Items (Not from Roblox)")
        
        # Filter out Roblox items (case-insensitive)
        # Handle NaN values and ensure creatorName is string
        df_not_roblox = df[
            df['creatorName'].notna() & 
            (df['creatorName'].astype(str).str.lower() != 'roblox')
        ].copy()
        
        if len(df_not_roblox) > 0:
            top_100_not_roblox = df_not_roblox.head(100)
            
            st.markdown(f"**Total Non-Roblox Assets:** {len(df_not_roblox)} | **Showing Top 100**")
            st.markdown("---")
            
            # Create display table for non-Roblox items
            display_df_not_roblox = top_100_not_roblox[["assetName", "creatorName", "timesSeen", "assetId"]].copy()
            display_df_not_roblox.columns = ["Asset Name", "Creator", "Times Seen", "Asset ID"]
            
            # Format the display
            display_df_not_roblox["Times Seen"] = display_df_not_roblox["Times Seen"].astype(int)
            display_df_not_roblox["Asset ID"] = display_df_not_roblox["Asset ID"].astype(int)
            
            # Display table
            st.dataframe(
                display_df_not_roblox,
                use_container_width=True,
                hide_index=True
            )
            
            # Summary statistics for non-Roblox items
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Non-Roblox Assets", f"{len(df_not_roblox):,}")
            with col2:
                st.metric("Total Times Seen", f"{int(df_not_roblox['timesSeen'].sum()):,}")
            with col3:
                st.metric("Average Times Seen", f"{df_not_roblox['timesSeen'].mean():.2f}")
            with col4:
                st.metric("Most Popular Item", f"{int(df_not_roblox.iloc[0]['timesSeen'])}" if len(df_not_roblox) > 0 else "N/A")
        else:
            st.info("No items found from creators other than Roblox.")
