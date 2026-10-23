import { useTopRecommendations, useSafeRecommendations, useAggressiveRecommendations, useComfortRecommendations, useAntiMetaRecommendations, useIsProcessing } from '../stores';
import { RecommendationCard } from './RecommendationCard';
import { motion } from 'framer-motion';

interface RecommendationCategoryProps {
  title: string;
  recommendations: Array<{
    hero_id: number;
    score: number;
    confidence: number;
    explanations: string[];
    risk_level: 'low' | 'medium' | 'high';
    role_fit: string[];
    category: 'top' | 'safe' | 'aggressive' | 'comfort' | 'anti_meta';
  }>;
  icon: React.ReactNode;
  emptyMessage: string;
}

function RecommendationCategory({ 
  title, 
  recommendations, 
  icon, 
  emptyMessage 
}: RecommendationCategoryProps) {
  if (recommendations.length === 0) {
    return (
      <div className="py-4">
        <h4 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-dota-textMuted mb-2 px-2">
          {icon}
          {title}
        </h4>
        <div className="text-center py-6 text-dota-textMuted text-sm">
          {emptyMessage}
        </div>
      </div>
    );
  }

  return (
    <div className="py-2">
      <h4 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-dota-textMuted mb-2 px-2">
        {icon}
        {title}
      </h4>
      <div className="grid grid-cols-2 gap-2 px-2">
        {recommendations.map((rec, index) => (
          <motion.div
            key={`${rec.hero_id}-${rec.category}`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05, duration: 0.2 }}
          >
            <RecommendationCard recommendation={rec} />
          </motion.div>
        ))}
      </div>
    </div>
  );
}

export function RecommendationPanel() {
  const topPicks = useTopRecommendations();
  const safePicks = useSafeRecommendations();
  const aggressivePicks = useAggressiveRecommendations();
  const comfortPicks = useComfortRecommendations();
  const antiMetaPicks = useAntiMetaRecommendations();
  const isProcessing = useIsProcessing();

  if (isProcessing) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-3 border-dota-accent border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-dota-textMuted">Analyzing draft...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-3 space-y-4">
      <RecommendationCategory
        title="Top Picks"
        recommendations={topPicks}
        icon={<svg className="w-4 h-4 text-dota-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 6l-4 4-4-4M5 10h14M5 14h14" /></svg>}
        emptyMessage="No top picks available"
      />
      
      <RecommendationCategory
        title="Safe Picks"
        recommendations={safePicks}
        icon={<svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg>}
        emptyMessage="No safe picks available"
      />
      
      <RecommendationCategory
        title="High Impact"
        recommendations={aggressivePicks}
        icon={<svg className="w-4 h-4 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>}
        emptyMessage="No aggressive picks available"
      />
      
      <RecommendationCategory
        title="Comfort Picks"
        recommendations={comfortPicks}
        icon={<svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" /></svg>}
        emptyMessage="No comfort data available"
      />
      
      <RecommendationCategory
        title="Anti-Meta"
        recommendations={antiMetaPicks}
        icon={<svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /></svg>}
        emptyMessage="No anti-meta picks available"
      />
    </div>
  );
}
