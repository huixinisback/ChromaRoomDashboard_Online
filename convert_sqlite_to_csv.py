"""
Convert SQLite datastore to CSV files.
Extracts data_raw JSON from each datastore_name and saves as separate CSV files.
"""

import json
import os
import sqlite3
import pandas as pd


def find_datastore_file():
    """Find the consolidated datastore.sqlite3 file."""
    datastore_files = []
    for f in os.listdir('.'):
        if f.startswith('datastore') and f.endswith('.sqlite3'):
            datastore_files.append(f)
    
    if not datastore_files:
        print("❌ No datastore.sqlite3 file found in current directory.")
        return None
    
    # Use the most recent datastore file if multiple exist
    datastore_file = sorted(datastore_files)[-1]
    print(f"📁 Found datastore file: {datastore_file}")
    return datastore_file


def get_datastore_names(conn):
    """Get all unique datastore_name values from the database."""
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    if not tables:
        print("❌ No tables found in database.")
        return []
    
    table_name = tables[0][0]
    print(f"📊 Using table: {table_name}")
    
    # Get column names
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'datastore_name' not in columns:
        print("❌ 'datastore_name' column not found in database.")
        return []
    
    # Get unique datastore names
    cursor.execute(f"SELECT DISTINCT datastore_name FROM {table_name}")
    datastore_names = [row[0] for row in cursor.fetchall()]
    
    return datastore_names


def extract_data_from_datastore(conn, table_name, datastore_name):
    """Extract and parse data_raw JSON for a specific datastore."""
    # Filter by datastore_name
    query = f"SELECT data_raw FROM {table_name} WHERE datastore_name = ?"
    df = pd.read_sql_query(query, conn, params=(datastore_name,))
    
    if df.empty:
        print(f"  ⚠️  No data found for {datastore_name}")
        return None
    
    # Parse JSON from data_raw column
    records = []
    for idx, row in df.iterrows():
        try:
            data_str = row['data_raw']
            if isinstance(data_str, str):
                obj = json.loads(data_str)
                records.append(obj)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"  ⚠️  Error parsing row {idx}: {e}")
            continue
    
    if not records:
        print(f"  ⚠️  No valid JSON records found for {datastore_name}")
        return None
    
    # Convert to DataFrame
    result_df = pd.DataFrame.from_records(records)
    return result_df


def convert_datastore_to_csv(datastore_file):
    """Convert all datastores in SQLite file to CSV files."""
    try:
        conn = sqlite3.connect(datastore_file)
        
        # Get table name
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        if not tables:
            print("❌ No tables found in database.")
            conn.close()
            return
        
        table_name = tables[0][0]
        
        # Get all unique datastore names
        datastore_names = get_datastore_names(conn)
        
        if not datastore_names:
            print("❌ No datastore names found.")
            conn.close()
            return
        
        print(f"\n📋 Found {len(datastore_names)} datastore(s):")
        for name in datastore_names:
            print(f"  - {name}")
        
        print("\n🔄 Converting datastores to CSV...\n")
        
        # Convert each datastore to CSV
        success_count = 0
        for datastore_name in datastore_names:
            print(f"📝 Processing {datastore_name}...")
            
            # Extract data
            df = extract_data_from_datastore(conn, table_name, datastore_name)
            
            if df is not None and not df.empty:
                # Create CSV filename (sanitize datastore name)
                csv_filename = f"{datastore_name}.csv"
                
                # Save to CSV
                df.to_csv(csv_filename, index=False)
                print(f"  ✅ Saved {len(df)} records to {csv_filename}")
                success_count += 1
            else:
                print(f"  ❌ Failed to extract data for {datastore_name}")
        
        conn.close()
        
        print(f"\n✨ Conversion complete! {success_count}/{len(datastore_names)} datastores converted successfully.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    print("=" * 60)
    print("🔄 SQLite to CSV Converter")
    print("=" * 60)
    print()
    
    # Find datastore file
    datastore_file = find_datastore_file()
    
    if datastore_file is None:
        return
    
    # Convert to CSV
    convert_datastore_to_csv(datastore_file)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

