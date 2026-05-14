# SerpAPI Connection Troubleshooting

## Issue
Your network cannot resolve `serpapi.com` DNS:
```
❌ DNS failed: [Errno 11001] getaddrinfo failed
```

## Root Cause
- **Corporate firewall** blocking access to serpapi.com
- **DNS filtering** preventing resolution
- **Proxy settings** not configured for external APIs

## Current Status
✅ **System is working** - using mock data fallback
❌ **Real job search unavailable** - cannot access Google Jobs API

## Solutions (in order of difficulty)

### Option 1: Use Mock Mode (Current Setup)
✅ **Already working** - system automatically falls back to mock data

**Pros:**
- Works immediately
- No configuration needed
- Good for testing/development

**Cons:**
- Mock jobs are samples, not real listings
- URLs point to mock career pages (realistic looking but not real)
- Limited job variety

**To use:** Just run your queries normally - mock mode is automatic when SerpAPI is unavailable.

---

### Option 2: Change DNS Server
Try using public DNS servers that don't block external APIs:

**Windows PowerShell (Run as Administrator):**
```powershell
# Set DNS to Google's public DNS
Set-DnsClientServerAddress -InterfaceIndex (Get-NetAdapter | Select-Object -First 1).ifIndex -ServerAddresses ("8.8.8.8","8.8.4.4")

# OR use Cloudflare DNS
Set-DnsClientServerAddress -InterfaceIndex (Get-NetAdapter | Select-Object -First 1).ifIndex -ServerAddresses ("1.1.1.1","1.0.0.1")

# Test serpapi.com resolution
Resolve-DnsName serpapi.com
```

**To revert to automatic DNS:**
```powershell
Set-DnsClientServerAddress -InterfaceIndex (Get-NetAdapter | Select-Object -First 1).ifIndex -ResetServerAddresses
```

---

### Option 3: Use VPN
If corporate firewall is blocking, try:
1. Connect to a personal VPN
2. Test connection: `python test_serpapi_connection.py`
3. Restart your app if DNS now works

---

### Option 4: Proxy Configuration
If your network requires a proxy:

**Add to your code (in `tools/job_search.py`):**
```python
async with aiohttp.ClientSession(
    connector=aiohttp.TCPConnector(
        ssl=False,  # If SSL is blocked
    ),
    proxy="http://your-proxy:port"  # Your proxy URL
) as session:
    # ... rest of code
```

---

### Option 5: Use Different Network
Test from:
- Home network
- Mobile hotspot
- Different office location

Run test: `python test_serpapi_connection.py`

---

### Option 6: Alternative Job APIs
If SerpAPI remains blocked, consider alternatives:

1. **LinkedIn API** (requires OAuth)
2. **Indeed API** (free tier available)
3. **Adzuna API** (free tier: 250 calls/month)
4. **TheJobOverflow API** (GitHub jobs)

Would require code changes to integrate.

---

## Testing
Always test after changes:
```powershell
python test_serpapi_connection.py
```

**Expected output when working:**
```
✅ Resolved to: 104.18.31.34
Status: 200
✅ Connection successful!
Search results: X jobs found
```

---

## Current Mock Data
Mock mode provides:
- 5 sample job listings
- Realistic company names (TechCorp, StartupXYZ, AI Innovations, etc.)
- Realistic career page URLs (e.g., `https://careers.techcorp.in/...`)
- Filtered by query keywords
- Skills matching

**Mock jobs include:**
1. Senior Python Developer - TechCorp India - Delhi
2. Backend Engineer - StartupXYZ - Bangalore
3. ML Engineer - AI Innovations - Bangalore  
4. Full Stack Developer - ProductCo - Delhi NCR
5. DevOps Engineer - CloudTech Solutions - Mumbai

---

## When Real API Works
Once DNS resolves successfully:
- System automatically switches to Google Jobs API
- Real job listings with actual company URLs
- Up to 100 results per search
- Fresh job postings
- Direct application links
- Real salary data

No code changes needed - automatic fallback!

---

## Questions?
- Mock mode: Already working ✅
- Need real jobs: Fix DNS/network access
- Want different API: Let me know which one

Run `python test_serpapi_connection.py` to check status anytime.
