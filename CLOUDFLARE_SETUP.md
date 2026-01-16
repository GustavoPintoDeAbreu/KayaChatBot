# Cloudflare Tunnel Setup Guide for KayaChatBot

This guide walks you through setting up Cloudflare Tunnel to make your KayaChatBot accessible via **sigmakayachatbot.pt**.

## Prerequisites

- Active Cloudflare account
- Domain **sigmakayachatbot.pt** added to your Cloudflare account
- Docker and docker-compose installed
- KayaChatBot app running locally

## Step-by-Step Setup

### 1. Add Domain to Cloudflare

1. Log in to [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Click **Add site** (if not already added)
3. Enter **sigmakayachatbot.pt**
4. Select the Free plan
5. Update your domain's nameservers at your registrar to the ones provided by Cloudflare
6. Wait for DNS propagation (can take up to 24 hours, usually much faster)

### 2. Create Cloudflare Tunnel

1. Navigate to **Zero Trust** dashboard: https://one.dash.cloudflare.com/
2. Go to **Networks** → **Tunnels**
3. Click **Create a tunnel**
4. Select **Cloudflared** as the connector type
5. Name your tunnel: `kaya-chatbot`
6. Click **Save tunnel**

### 3. Configure Public Hostname

In the tunnel configuration:

1. **Public Hostname** tab:
   - **Subdomain**: Leave empty (or use `www` if you prefer www.sigmakayachatbot.pt)
   - **Domain**: Select `sigmakayachatbot.pt`
   - **Path**: Leave empty
   
2. **Service** configuration:
   - **Type**: `HTTP`
   - **URL**: `frontend:3000`
   
   > **Important**: Use `frontend:3000`, NOT `localhost:3000`. This is the Docker internal service name.

3. Click **Save hostname**

### 4. Get Tunnel Token

1. After saving, you'll see your tunnel in the list
2. Click the **three dots menu (⋯)** → **Configure**
3. Scroll down to find the **Install and run a connector** section
4. Copy the command shown. It will look like:
   ```bash
   cloudflared tunnel run --token eyJhIjoiX...long_token_here...fQ
   ```
5. **Copy only the token part** (everything after `--token `)

### 5. Add Token to .env File

1. Open `.env` file in your KayaChatBot directory
2. Find the line:
   ```env
   # CLOUDFLARE_TOKEN=your_cloudflare_tunnel_token_here
   ```
3. Uncomment and replace with your actual token:
   ```env
   CLOUDFLARE_TOKEN=eyJhIjoiX...your_actual_token_here...fQ
   ```
4. Save the file

### 6. Update CORS Settings (Optional but Recommended)

For production security, update API CORS settings:

1. Open `src/api/main.py`
2. Find the CORS configuration (around line 20):
   ```python
   origins = ["*"]
   ```
3. Update to your specific domain:
   ```python
   origins = [
       "https://sigmakayachatbot.pt",
       "http://localhost:3000",  # Keep for local testing
   ]
   ```

### 7. Start Services with Tunnel

Run the following commands in your project directory:

```powershell
# Start all services including Cloudflare tunnel
docker-compose up -d --build
```

This will start:
- ✅ Ollama (model server)
- ✅ API (FastAPI backend)
- ✅ Frontend (React app)
- ✅ Cloudflared (tunnel connector)

### 8. Verify Tunnel Connection

1. Check tunnel status in Cloudflare dashboard:
   - Go to **Zero Trust** → **Networks** → **Tunnels**
   - Your tunnel should show as **HEALTHY** (green status)

2. Check Docker logs:
   ```powershell
   docker-compose logs cloudflared
   ```
   You should see: `Connection <id> registered connIndex=0`

### 9. Test Your App

1. **Local access**: http://localhost:3000
2. **Public access**: https://sigmakayachatbot.pt

Both should show the KayaChatBot login page.

## DNS Verification

If your domain doesn't work immediately:

1. Go to Cloudflare Dashboard → **DNS** → **Records**
2. Verify you have a CNAME record:
   - **Type**: CNAME
   - **Name**: `@` (or `www`)
   - **Target**: `<tunnel-id>.cfargotunnel.com`
   - **Proxy status**: Proxied (orange cloud)

This record is usually created automatically by Cloudflare Tunnel.

## Access Control

Your app uses email-based authentication. To allow friends to access:

1. Open `.env`
2. Update `ALLOWED_EMAILS`:
   ```env
   ALLOWED_EMAILS=gukler@gmail.com,friend1@example.com,friend2@example.com
   ```
3. Restart the API service:
   ```powershell
   docker-compose restart api
   ```

## Troubleshooting

### Tunnel shows as UNHEALTHY

**Problem**: Tunnel status is red/unhealthy in Cloudflare dashboard

**Solutions**:
- Check if cloudflared container is running: `docker ps`
- View logs: `docker-compose logs cloudflared`
- Verify token is correct in `.env`
- Restart tunnel: `docker-compose restart cloudflared`

### Can access localhost but not public URL

**Problem**: http://localhost:3000 works but https://sigmakayachatbot.pt doesn't

**Solutions**:
- Wait for DNS propagation (can take up to 24 hours)
- Clear browser cache and try incognito mode
- Check tunnel is HEALTHY in Cloudflare dashboard
- Verify public hostname is configured correctly (Service should be `frontend:3000`)
- Try: `nslookup sigmakayachatbot.pt` to verify DNS

### 502 Bad Gateway Error

**Problem**: Public URL shows "502 Bad Gateway"

**Solutions**:
- Verify all services are running: `docker ps`
- Check frontend service specifically: `docker-compose logs frontend`
- Ensure service URL in tunnel config is `frontend:3000` (not `localhost:3000`)
- Restart all services: `docker-compose restart`

### Authentication doesn't work

**Problem**: Can't log in with email

**Solutions**:
- Verify email is in `ALLOWED_EMAILS` in `.env`
- Check `SECRET_KEY` is set in `.env`
- Restart API service: `docker-compose restart api`
- Check API logs: `docker-compose logs api`

### CORS Errors in Browser Console

**Problem**: Browser shows CORS policy errors

**Solutions**:
- Update `origins` in `src/api/main.py` to include `https://sigmakayachatbot.pt`
- Rebuild API container: `docker-compose up -d --build api`
- Clear browser cache

## Security Best Practices

1. **Keep SECRET_KEY secure**: Never commit it to git
2. **Update ALLOWED_EMAILS**: Only add trusted friends
3. **Monitor access logs**: Check `docker-compose logs api` regularly
4. **Use HTTPS only**: Cloudflare provides this automatically
5. **Enable Cloudflare features**:
   - Go to Security → WAF → Enable firewall rules
   - Enable **Bot Fight Mode** under Security → Bots
   - Consider enabling **Rate Limiting**

## Updating the App

When you make changes to your code:

```powershell
# Rebuild and restart all services
docker-compose up -d --build

# Or rebuild specific service
docker-compose up -d --build api
docker-compose up -d --build frontend
```

## Stopping the App

```powershell
# Stop all services
docker-compose down

# Stop and remove all data (including volumes)
docker-compose down -v
```

## Cost Estimate

- **Cloudflare Tunnel**: FREE
- **Cloudflare DNS**: FREE
- **Domain registration**: Varies by registrar (~$10-20/year for .pt domains)
- **Hosting**: FREE (running on your local machine)

## Next Steps

1. ✅ Complete tunnel setup using this guide
2. ✅ Test local access (localhost:3000)
3. ✅ Test public access (sigmakayachatbot.pt)
4. ✅ Add friends' emails to `ALLOWED_EMAILS`
5. ✅ Share link with friends!

## Support

If you encounter issues:
1. Check this troubleshooting section
2. Review Docker logs: `docker-compose logs`
3. Verify Cloudflare tunnel status in dashboard
4. Check Cloudflare documentation: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/

---

**Ready to deploy?** Follow the steps above and your KayaChatBot will be live at https://sigmakayachatbot.pt! 🚀
