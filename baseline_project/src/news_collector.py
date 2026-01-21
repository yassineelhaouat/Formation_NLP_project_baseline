# src/news_collector.py
"""
📥 Module Collecte d'Articles

Collecte articles de sources multiples :
- HackerNews (BeautifulSoup)
- RSS feeds (feedparser)
- YouTube (optionnel, API)

Usage:
    collector = NewsCollector(config)
    articles = collector.collect_all()
"""

import requests
from bs4 import BeautifulSoup
import feedparser
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

# ═══════════════════════════════════════════════════════════════════════════
# LOGGER
# ═══════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Article:
    """Représentation standardisée d'un article"""
    title: str
    url: str
    content: str
    source: str
    date: Optional[str] = None
    author: Optional[str] = None
    score: Optional[int] = None  # Pour HN: points/upvotes
    
    def to_dict(self):
        return asdict(self)

# ═══════════════════════════════════════════════════════════════════════════
# NEWS COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════

class NewsCollector:
    """Collecte articles de multiples sources"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Dict de configuration (voir config.json)
        """
        self.config = config
        self.articles: List[Article] = []
        self.errors: List[Dict] = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; NewsCollector/1.0)'
        })
    
    # ═══════════════════════════════════════════════════════════════════════
    # HACKER NEWS
    # ═══════════════════════════════════════════════════════════════════════

    def fetch_article_content(self, url: str, timeout: int = 10) -> str:
        """
        Récupérer le contenu textuel d'un article depuis son URL
        
        Args:
            url: URL de l'article
            timeout: Délai maximum
        
        Returns:
            Contenu textuel extrait (premiers 1000 caractères)
        """
        try:
            # Ignorer certains domaines problématiques
            skip_domains = ['youtube.com', 'twitter.com', 'x.com', 'github.com', 'pdf']
            if any(domain in url.lower() for domain in skip_domains):
                return ""
            
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Supprimer scripts, styles, nav, footer
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Extraire texte des paragraphes
            paragraphs = soup.find_all('p')
            content = ' '.join(p.get_text(strip=True) for p in paragraphs[:10])
            
            # Limiter à 1000 caractères
            return content if content else ""
        
        except Exception as e:
            logger.debug(f"Could not fetch content from {url}: {str(e)}")
            return ""

    def collect_from_hacker_news(self) -> List[Article]:
        """
        Scrape Hacker News avec récupération du contenu
        """
        logger.info("🔄 Collecte HackerNews...")
        articles = []
        
        try:
            config = self.config.get('collection', {}).get('hacker_news', {})
            url = config.get('url', 'https://news.ycombinator.com/')
            num_pages = config.get('num_pages', 2)
            timeout = config.get('timeout', 10)
            delay = config.get('delay_between_requests', 2)
            fetch_content = config.get('fetch_content', False)  # NOUVEAU
            
            for page_num in range(num_pages):
                page_url = f"{url}?p={page_num + 1}" if page_num > 0 else url
                
                logger.info(f"  Scraping page {page_num + 1}/{num_pages}")
                
                try:
                    response = self.session.get(page_url, timeout=timeout)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    rows = soup.find_all('tr', class_='athing')
                    
                    for row in rows:
                        try:
                            # Extraire titre et URL
                            title_link = row.find('span', class_='titleline').find('a')
                            if not title_link:
                                continue
                            
                            title = title_link.get_text(strip=True)
                            article_url = title_link.get('href', '')
                            
                            # Extraire metadata
                            subtext_row = row.find_next('tr', class_='subtext')
                            if subtext_row:
                                score_span = subtext_row.find('span', class_='score')
                                score = int(score_span.get_text().split()[0]) if score_span else None
                                
                                author_link = subtext_row.find('a', class_='hnuser')
                                author = author_link.get_text() if author_link else None
                                
                                age_span = subtext_row.find('span', class_='age')
                                date = age_span.get('title') if age_span else None
                            else:
                                score = author = date = None
                            
                            # NOUVEAU: Récupérer contenu réel si activé
                            content = title  # Default: titre
                            if fetch_content and article_url.startswith('http'):
                                fetched = self.fetch_article_content(article_url, timeout=5)
                                if fetched:
                                    content = f"{title}. {fetched}"
                                    logger.debug(f"    ✓ Content fetched for: {title[:30]}...")
                            
                            article = Article(
                                title=title,
                                url=article_url,
                                content=content,  # AMÉLIORÉ
                                source='HackerNews',
                                date=date,
                                author=author,
                                score=score
                            )
                            
                            articles.append(article)
                        
                        except Exception as e:
                            self.errors.append({
                                'source': 'HackerNews',
                                'error': str(e),
                                'timestamp': datetime.now().isoformat()
                            })
                            continue
                    
                    time.sleep(delay)
                
                except requests.RequestException as e:
                    self.errors.append({
                        'source': 'HackerNews',
                        'page': page_num + 1,
                        'error': f"Request failed: {str(e)}",
                        'timestamp': datetime.now().isoformat()
                    })
                    logger.warning(f"  ✗ Erreur page {page_num + 1}: {str(e)}")
            
            logger.info(f"✅ HackerNews: {len(articles)} articles collectés")
            return articles
        
        except Exception as e:
            logger.error(f"❌ HackerNews collection failed: {str(e)}")
            return []

    # ═══════════════════════════════════════════════════════════════════════
    # TOWARDS DATA SCIENCE (HTML SCRAPING)
    # ═══════════════════════════════════════════════════════════════════════
    
    def collect_from_towards_data_science(self) -> List[Article]:
        """
        Scrape Towards Data Science /latest avec pagination
        
        URLs:
        - https://towardsdatascience.com/latest
        - https://towardsdatascience.com/latest/page/2
        - etc.
        """
        logger.info("📰 Collecte Towards Data Science...")
        articles = []
        
        try:
            config = self.config.get('collection', {}).get('towards_data_science', {})
            base_url = config.get('url', 'https://towardsdatascience.com/latest')
            num_pages = config.get('num_pages', 2)
            timeout = config.get('timeout', 15)
            delay = config.get('delay_between_requests', 2)
            fetch_content = config.get('fetch_content', True)
            
            for page_num in range(1, num_pages + 1):
                if page_num == 1:
                    page_url = base_url
                else:
                    page_url = f"{base_url}/page/{page_num}"
                
                logger.info(f"  Scraping TDS page {page_num}/{num_pages}: {page_url}")
                
                try:
                    response = self.session.get(page_url, timeout=timeout)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # TDS uses article cards - find all article links
                    # Multiple possible selectors based on their structure
                    article_links = soup.find_all('a', href=True)
                    
                    seen_urls = set()
                    
                    for link in article_links:
                        try:
                            href = link.get('href', '')
                            
                            # Filter for article URLs (they contain the article slug)
                            # TDS articles: /title-of-article-abc123def456
                            if not href.startswith('https://towardsdatascience.com/'):
                                continue
                            
                            # Skip non-article pages (author pages, tags, etc.)
                            skip_patterns = ['/latest', '/tag/', '/search', '/about', 
                                           '/archive', '/login', '/signup', '/@', 
                                           '/membership', '/plans', '/topics',
                                           '/author/', '/tagged/', '/collection/']
                            if any(pattern in href for pattern in skip_patterns):
                                continue
                            
                            # Skip if already seen
                            if href in seen_urls:
                                continue
                            seen_urls.add(href)
                            
                            # Get title from link text or nested elements
                            title = link.get_text(strip=True)
                            if not title or len(title) < 10:
                                # Try to find h2/h3 child
                                h_tag = link.find(['h1', 'h2', 'h3'])
                                if h_tag:
                                    title = h_tag.get_text(strip=True)
                            
                            if not title or len(title) < 10:
                                continue
                            
                            # Skip if title looks like navigation or author name (short, no spaces)
                            if title.lower() in ['read more', 'continue reading', 'see more']:
                                continue
                            
                            # Skip titles that look like author names (2-3 words, capitalized)
                            words = title.split()
                            if len(words) <= 3 and all(w[0].isupper() for w in words if w):
                                continue
                            
                            # Fetch content if enabled
                            content = title
                            if fetch_content:
                                logger.debug(f"    Fetching: {title[:40]}...")
                                fetched = self.fetch_article_content(href, timeout=10)
                                if fetched:
                                    content = f"{title}. {fetched}"
                                time.sleep(delay * 0.5)  # Shorter delay for content fetch
                            
                            article = Article(
                                title=title,
                                url=href,
                                content=content,
                                source='TowardsDataScience',
                                date=datetime.now().isoformat(),
                                author=None
                            )
                            
                            articles.append(article)
                            logger.debug(f"    ✓ {title[:50]}...")
                        
                        except Exception as e:
                            self.errors.append({
                                'source': 'TowardsDataScience',
                                'error': str(e),
                                'timestamp': datetime.now().isoformat()
                            })
                            continue
                    
                    logger.info(f"    Found {len(seen_urls)} articles on page {page_num}")
                    time.sleep(delay)
                
                except requests.RequestException as e:
                    self.errors.append({
                        'source': 'TowardsDataScience',
                        'page': page_num,
                        'error': f"Request failed: {str(e)}",
                        'timestamp': datetime.now().isoformat()
                    })
                    logger.warning(f"  ✗ Erreur page {page_num}: {str(e)}")
            
            logger.info(f"✅ TowardsDataScience: {len(articles)} articles collectés")
            return articles
        
        except Exception as e:
            logger.error(f"❌ TowardsDataScience collection failed: {str(e)}")
            return []

    
    # ═══════════════════════════════════════════════════════════════════════
    # RSS FEEDS
    # ═══════════════════════════════════════════════════════════════════════
    
    def collect_from_rss_feed(self, feed_url: str, max_articles: int = None, fetch_content: bool = False, delay: float = 1.0) -> List[Article]:
        """
        Parse RSS feed
        
        Args:
            feed_url: URL du flux RSS
            max_articles: Nombre max d'articles à collecter (None = tous)
            fetch_content: Si True, récupère le contenu complet de l'article
            delay: Délai entre requêtes (si fetch_content=True)
        
        Structure RSS standard :
        <rss>
          <channel>
            <item>
              <title>Article Title</title>
              <link>https://example.com/article</link>
              <description>Article summary</description>
              <pubDate>Sun, 15 Jan 2025 10:30:00 GMT</pubDate>
              <author>john@example.com</author>
            </item>
          </channel>
        </rss>
        """
        logger.info(f"🔄 Collecte RSS: {feed_url}")
        if max_articles:
            logger.info(f"   Max articles: {max_articles}")
        articles = []
        
        try:
            feed = feedparser.parse(feed_url)
            
            if feed.bozo:
                logger.warning(f"  ⚠️ RSS parsing warning: {feed.bozo_exception}")
            
            entries_to_process = feed.entries
            if max_articles:
                entries_to_process = feed.entries[:max_articles]
            
            for i, entry in enumerate(entries_to_process):
                try:
                    title = entry.get('title', 'No Title')
                    link = entry.get('link', '')
                    summary = entry.get('summary', '') or entry.get('description', '')
                    
                    # Parser date
                    date = None
                    if 'published' in entry:
                        date = entry.published
                    elif 'updated' in entry:
                        date = entry.updated
                    
                    author = None
                    if 'author' in entry:
                        author = entry.author
                    
                    # Fetch full content if enabled
                    content = summary[:200]  # Default: premiers 200 chars du summary
                    if fetch_content and link:
                        logger.info(f"   [{i+1}/{len(entries_to_process)}] Fetching: {title[:40]}...")
                        full_content = self.fetch_article_content(link)
                        if full_content:
                            content = full_content
                        time.sleep(delay)  # Respecter rate limiting
                    
                    article = Article(
                        title=title,
                        url=link,
                        content=content,
                        source='RSS',
                        date=date,
                        author=author
                    )
                    
                    articles.append(article)
                    logger.debug(f"    ✓ {title[:50]}...")
                
                except Exception as e:
                    self.errors.append({
                        'source': 'RSS',
                        'feed': feed_url,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    continue
            
            logger.info(f"✅ RSS: {len(articles)} articles collectés")
            return articles
        
        except Exception as e:
            logger.error(f"❌ RSS collection failed: {str(e)}")
            self.errors.append({
                'source': 'RSS',
                'feed': feed_url,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return []
    
    # ═══════════════════════════════════════════════════════════════════════
    # YOUTUBE (OPTIONNEL)
    # ═══════════════════════════════════════════════════════════════════════
    
    def collect_from_youtube(self) -> List[Article]:
        """
        Récupère métadonnées vidéos YouTube
        
        NOTE: Nécessite API key YouTube
        Pour tester sans : retourner liste vide
        """
        logger.info("🔄 YouTube (optionnel, non-implémenté dans baseline)")
        return []
    
    # ═══════════════════════════════════════════════════════════════════════
    # COLLECTE TOUS SOURCES
    # ═══════════════════════════════════════════════════════════════════════
    
    def collect_all(self) -> List[Article]:
        """Collecte tous sources activées"""
        logger.info("=" * 70)
        logger.info("📥 COLLECTE ARTICLES - Début")
        logger.info("=" * 70)
        
        self.articles = []
        
        # HackerNews
        if self.config.get('collection', {}).get('hacker_news', {}).get('enabled', True):
            self.articles.extend(self.collect_from_hacker_news())
        
        # Towards Data Science (HTML scraping)
        if self.config.get('collection', {}).get('towards_data_science', {}).get('enabled', False):
            self.articles.extend(self.collect_from_towards_data_science())
        
        # RSS - PyCoder's Weekly (disabled by default)
        if self.config.get('collection', {}).get('pycoders_rss', {}).get('enabled', False):
            rss_url = self.config['collection']['pycoders_rss'].get('url')
            if rss_url:
                self.articles.extend(self.collect_from_rss_feed(rss_url))
        
        # YouTube (optional)
        if self.config.get('collection', {}).get('youtube', {}).get('enabled', False):
            self.articles.extend(self.collect_from_youtube())
        
        logger.info("=" * 70)
        logger.info(f"📊 RÉSUMÉ COLLECTE")
        logger.info(f"   Total articles : {len(self.articles)}")
        logger.info(f"   Total erreurs : {len(self.errors)}")
        logger.info("=" * 70)
        
        return self.articles
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAUVEGARDE DONNÉES
    # ═══════════════════════════════════════════════════════════════════════
    
    def save_to_jsonl(self, filepath: str):
        """Sauvegarde articles en format JSONL"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for article in self.articles:
                    f.write(json.dumps(article.to_dict(), ensure_ascii=False) + '\n')
            logger.info(f"✅ Articles sauvegardés: {filepath}")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {str(e)}")
    
    def save_errors_log(self, filepath: str):
        """Sauvegarde log d'erreurs"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.errors, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Erreurs sauvegardées: {filepath}")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde log: {str(e)}")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN - TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    # Charger config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Collecter
    collector = NewsCollector(config)
    articles = collector.collect_all()
    
    # Sauvegarder
    collector.save_to_jsonl('data/articles_raw.jsonl')
    collector.save_errors_log('data/collection_errors.json')
    
    # Afficher sample
    if articles:
        print("\n📄 Premier article :")
        print(f"  Titre: {articles[0].title}")
        print(f"  URL: {articles[0].url}")
        print(f"  Source: {articles[0].source}")
